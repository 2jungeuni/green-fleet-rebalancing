# Built-in
import os
import sys
import csv
import math
import pickle
import numpy as np
from pathlib import Path
from tabulate import tabulate
from collections import deque
from datetime import datetime
from scipy.spatial import KDTree

# Logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Own
from config import cfg
from opt.opt import solve
from utils.utils import *
from c_backend import engine
from framework.framework import *
from decoder.dqn import DQNAgent
from decoder.td3 import TD3Agent
from utils.args import parse_args
from utils.logging import CSVLogger
from encoder.stgcn.base import STGCN
from encoder.stgcn.trainer import STGCNTrainer
from decoder.qmix_m import QMIXAgent as QMIXMAgent
from decoder.qmix_a import QMIXAgent as QMIXAAgent

# SUMO / Traci
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import traci
import sumolib
from traci import TraCIException

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Global sets for reservations, vehicles, etc.
Reservations: Set[Reservation] = set()
Vehicles: Set[Vehicle] = set()

Pickup: Set[NodeObject] = set()
Serviced: Set[NodeObject] = set()
Rejected: Set[NodeObject] = set()

# Dictionary to track when each vehicle has been empty since
VehicleEmpty: Dict[str, int] = {}
EmptyCount: Dict[str, int] = {}

# For representing solutions
PUOrder = List[int]
DOOrder = List[int]
Solution = Dict[int, Tuple[PUOrder, DOOrder]]
CarbonEmission = Dict[int, float]
LastDispatchedStops: Dict[str, List[str]] = {}

# -------------------------------------------------
# Global indices
# -------------------------------------------------
# Mapping from a (type, key) to a global node index
GLOBAL_NODE_INDEX: Dict[Tuple[str, str], int] = {}
GLOBAL_OBJECTS: Dict[int, NodeObject] = {}

# Depot key
DEPOT_KEY = ("depot", "global_depot")

torch.autograd.set_detect_anomaly(True)

# Find nearest edges
def find_nearest_edge_kdtree(x, y, edge_dict):
    ids, coords = zip(*edge_dict.items())
    tree = KDTree(coords)
    _, idx = tree.query((x, y))
    # print("Edge ID: ", ids[idx], "Coordinate: ", coords[idx])
    return ids[idx]

# Global index for the depot, Creates new if not existing
def get_or_create_depot_index() -> int:
    if DEPOT_KEY not in GLOBAL_NODE_INDEX:
        new_idx = len(GLOBAL_OBJECTS)
        GLOBAL_NODE_INDEX[DEPOT_KEY] = new_idx
        GLOBAL_OBJECTS[new_idx] = "depot"
    return GLOBAL_NODE_INDEX[DEPOT_KEY]

# Assign global index to either the pickup-node or dropoff-node of reservation
def get_or_create_reservation_index(r: Reservation, is_from: bool) -> int:
    if is_from:
        key = ("res_from", r.get_persons()[0])
    else:
        key = ("res_to", r.get_persons()[0])

    if key in GLOBAL_NODE_INDEX:
        return GLOBAL_NODE_INDEX[key]

    new_idx = len(GLOBAL_OBJECTS)
    GLOBAL_NODE_INDEX[key] = new_idx
    GLOBAL_OBJECTS[new_idx] = r
    return new_idx

# Similar to reservation index, but for vehicles
def get_or_create_vehicle_index(v: Vehicle) -> int:
    key = ("veh", v.id_vehicle)

    if key in GLOBAL_NODE_INDEX:
        return GLOBAL_NODE_INDEX[key]

    new_idx = len(GLOBAL_OBJECTS)
    GLOBAL_NODE_INDEX[key] = new_idx
    GLOBAL_OBJECTS[new_idx] = v
    return new_idx

def revise_solution(solution: Solution, r: Reservation):
    for veh_id in solution:
        pickups, dropoffs = solution[veh_id]
        revised_pickups, revised_dropoffs = None, None
        if r.from_node in pickups:
            revised_pickups = pickups.remove(r.from_node)
        if r.to_node in dropoffs:
            revised_dropoffs = dropoffs.remove(r.to_node)

        if revised_pickups:
            solution[veh_id][0] = revised_pickups
        if revised_dropoffs:
            solution[veh_id][1] = revised_dropoffs
        if revised_pickups or revised_dropoffs:
            break
    return solution

def create_nodes(
        reservations: Set[Reservation],
        vehicles: Set[Vehicle],
) -> Dict[int, NodeObject]:
    """
    Create or retrieve global indices (and objects) for depot, reservations, and vehicles.
    Build a node_dict that maps each valid index to the corresponding object.
    """
    # Depot index
    depot_idx = get_or_create_depot_index()

    # Assign from_node / to_node index for each reservation
    for r in reservations:
        if not r.is_picked_up():
            r.from_node = get_or_create_reservation_index(r, True)
            r.to_node = get_or_create_reservation_index(r, False)
        else:
            r.to_node = get_or_create_reservation_index(r, False)

    # Assign indices for each vehicle
    for v in vehicles:
        _ = get_or_create_vehicle_index(v)

    # Filter out only valid indices for constructing final node_dict
    valid_indices: Set[int] = set()
    valid_indices.add(depot_idx)

    for r in reservations:
        if r.from_node is not None:
            valid_indices.add(r.from_node)
        if r.to_node is not None:
            valid_indices.add(r.to_node)

    for v in vehicles:
        idx_v = GLOBAL_NODE_INDEX.get(("veh", v.id_vehicle), None)
        if idx_v is not None:
            valid_indices.add(idx_v)

    # Build the final dictionary
    node_dict: Dict[int, NodeObject] = {}
    for idx in valid_indices:
        node_dict[idx] = GLOBAL_OBJECTS[idx]

    return node_dict

# Creates / Updates global node indices for all reservations + vehicles + depot
def create_problem(
        solution: Solution,
        sumo_fleet: List[str],
        cost_type: CostType,
        waiting_time: int,
        travel_delay: int,
        end: int,
        start_time: int,
        timestep: float,
        verbose: bool
) -> Tuple[Framework, Solution]:
    """
    1) Identify newly created vehicles in SUMO and add them to the set Vehicles.
    2) Remove vehicles that no longer exist in SUMO.
    3) Synchronize reservations with SUMO.
    4) Reject reservations that exceed waiting time.
    5) Build objects and cost matrix.
    6) Build the Framework object for the optimization problem.
    """
    # 1) Check & create new vehicles
    previous_vehicles = [veh.id_vehicle for veh in Vehicles]
    for veh_id in sumo_fleet:
        if veh_id not in previous_vehicles:
            v = Vehicle(
                id_vehicle=veh_id,
                vehicle_index=int(veh_id.split("_")[1]),
                start_time=traci.simulation.getTime()
            )
            Vehicles.add(v)

    # 2) Update vehicle start_node if no passengers
    removed_vehicles = []
    for veh in Vehicles:
        # If the vehicle no longer exists in SUMO
        if veh.id_vehicle not in sumo_fleet:
            removed_vehicles.append(veh)

        # If vehicle has no passengers adn not in solution, reset start time & start node
        try:
            if len(list(traci.vehicle.getPersonIDList(veh.id_vehicle))) == 0 and  veh.vehicle_index not in solution:
                veh.start_time = round(traci.simulation.getTime())
                veh.start_node = veh.get_edge()
        except TraCIException:
            pass

    # Remove the solution info of vehicles that disappeared
    for v in removed_vehicles:
        Vehicles.remove(v)
        vid = v.vehicle_index
        if v in VehicleEmpty:
            del VehicleEmpty[v]
        if vid in solution:
            pickups, dropoffs = solution[vid]
            for req in dropoffs:
                Rejected.add(req)
            del solution[vid]

    # 3) Synchronize reservations with SUMO
    sumo_reservations = traci.person.getTaxiReservations(0)
    sumo_res_ids = [r.id for r in sumo_reservations]
    data_res_ids = [r.get_id() for r in Reservations]

    # Remove completed reservations from 'Reservations' and from 'solution'
    to_remove = []
    for r in Reservations:
        if r.get_id() not in sumo_res_ids:
            Serviced.add(r)
            to_remove.append(r)

    for r in to_remove:
        Reservations.remove(r)
        solution = revise_solution(solution, r)

    # Add new reservations
    for sr in sumo_reservations:
        if sr.id not in data_res_ids:
            Reservations.add(Reservation(sr, pickup_earliest=round(traci.simulation.getTime())))

    # 4) Reject reservations exceed waiting time
    rej_nodes = []
    ava_res, rej_res = reject_late_reservations(Reservations, Vehicles, waiting_time, timestep)
    for r in rej_res:
        if r in Reservations:
            Rejected.add(r)
            Reservations.remove(r)
            traci.person.remove(r.get_persons()[0])
            rej_nodes.append(r)

    if rej_nodes:
        for r in rej_nodes:
            solution = revise_solution(solution, r)

    # 5) Create node objects
    node_objects = create_nodes(Reservations, Vehicles)

    # 6) Build cost matrix & time matrix
    time_matrix, dist_matrix, cost_matrix = get_cost_matrix(node_objects, start_time, cost_type)

    # 7) Start nodes for each vehicle
    start_nodes = [v.start_node for v in Vehicles]

    # 8) Time windows
    time_windows = {}
    for idx, node in node_objects.items():
        tw = get_time_window_of_node_object(node, idx, end, start_time, waiting_time)
        time_windows[idx] = (tw)

    # Build framework
    data = Framework(
        depot=0,
        cost_matrix=cost_matrix,
        time_matrix=time_matrix,
        dist_matrix=dist_matrix,
        num_vehicles=len(Vehicles),
        starts=start_nodes,
        ends=[0] * len(Vehicles),
        waiting_time=waiting_time,
        travel_delay=travel_delay,
        time_windows=time_windows,
        max_time=end,
        node_objects=node_objects,
        reservations=Reservations,
        vehicles=Vehicles,
        cost_type=cost_type,
    )
    return data, solution

def dispatch(
        start_time: int,
        time_limit: int,
        data: Framework,
        solution: Solution,
        verbose: bool = False
) -> Solution:
    """
    1) Call C++ backend for feasible solutions
    2) Merge with existing solution, solve with Gurobi / Python
    3) Apply the new solution -> dispatch stops in SUMO
    """
    # 1) Prepare info for C++ engine
    res_info = []
    for r in data.reservations:
        from_idx = GLOBAL_NODE_INDEX.get(("res_from", r.get_persons()[0]), -1)
        to_idx = GLOBAL_NODE_INDEX.get(("res_to", r.get_persons()[0]), -1)
        if r.is_picked_up():
            res_info.append((-1, to_idx))
        else:
            res_info.append((from_idx, to_idx))

    veh_info = {}
    for v in data.vehicles:
        start_idx = GLOBAL_NODE_INDEX.get(("veh", v.id_vehicle), -1)
        end_idx = data.depot
        veh_info[v.vehicle_index] = (start_idx, end_idx)

    # 2) Call C++ engine
    feasible_sol = engine.run(
        res_info,
        veh_info,
        data.time_matrix,
        data.time_windows,
        solution,
        data.waiting_time,
        data.travel_delay
    )

    # 3) Remove -1 from both existing and feasible solutions
    for vid, (pu_list, do_list) in solution.items():
        new_pu = [nid for nid in pu_list if nid != -1]
        new_do = [nid for nid in do_list if nid != -1]
        solution[vid] = (new_pu, new_do)

    for i, sol_dict in enumerate(feasible_sol):
        for v, (pu_nodes, do_nodes) in sol_dict.items():
            # print("Veh: ", v, " / ", "PU: ", pu_nodes, " DO: ", do_nodes)
            filtered_pu = [n for n in pu_nodes if n != -1]
            filtered_do = [n for n in do_nodes if n != -1]
            feasible_sol[i] = {v: (filtered_pu, filtered_do)}

    # 4) Solve with Python (Gurobi) to get the best solution
    new_solution, obj_val = solve(
        solution,
        feasible_sol,
        data.vehicles,
        data.node_objects,
        data.time_matrix,
        data.dist_matrix
    )

    # 5) Dispatch new solution into SUMO
    for veh_index, (pickup_nodes, dropoff_nodes) in new_solution.items():
        veh = next((v for v in data.vehicles if v.vehicle_index == veh_index), None)
        if not veh:
            continue
        veh_id = veh.id_vehicle
        veh_type = veh.get_type_ID()

        # Update vehicle passenger count
        veh.num_passengers = len(dropoff_nodes)

        # Update each reservation's vehicle property if needed
        for nidx in dropoff_nodes:
            if nidx in data.node_objects:
                node = data.node_objects[nidx]
                if node.vehicle:
                    continue
                else:
                    node.vehicle = veh_id
                    node.update_direct_route_cost(veh_type, data.time_matrix, CostType.TIME)

        stops = []
        # Build pickup stops
        for nidx in pickup_nodes:
            if nidx in data.node_objects:
                node = data.node_objects[nidx]
                if isinstance(node, Reservation):
                    if node.is_picked_up():
                        continue
                    else:
                        stops.append(node.get_id())

        # Build dropoff stops
        for nidx in dropoff_nodes:
            if nidx in data.node_objects:
                node = data.node_objects[nidx]
                if isinstance(node, Reservation):
                    stops.append(node.get_id())
                else:
                    continue

        old_stops = LastDispatchedStops.get(veh_id, [])
        if stops == old_stops:
            # print(f"[SKIP] {veh_id} has stops same as before.")
            continue

        # Dispatch stops in SUMO
        if stops:
            traci.vehicle.dispatchTaxi(veh_id, stops)
            LastDispatchedStops[veh_id] = stops
        else:
            LastDispatchedStops[veh_id] = []

    return new_solution

# Retrieves SUMO's end time. If infinite, just return 90,000 as a safe bound
def get_max_time() -> int:
    max_sim_time = traci.simulation.getEndTime()
    return 90000 if max_sim_time == -1 else max_sim_time

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Main simulation loop that steps through SUMO time, calls dispatch when needed
def run(
        end: int = None,
        date: str = None,
        time_limit: float = 10,
        cost_type: CostType = CostType.TIME,
        waiting_time: int = 300,
        travel_delay: int = 420,
        encoder_trainer: STGCNTrainer = None,
        decoder_agent: DQNAgent = None,
        device: str = 'cpu',
        verbose: bool = False
):
    # Logger
    if cfg.version == "dqn":
        fieldnames = {
            "log": ["timestep", "vehicle id", "position", "person list", "carbon emission", "cumulative distance"],
            "plog": ["timestep", "person id", "waiting time"],
            "solution_log": ["timestep", "vehicle id", "sequence of pickup locations", "sequence of drop-off locations",
                             "travel delay"],
            "service_rate_log": ["timestep", "service rate"],
            "encoder_loss": ["timestep", "train loss", "copy loss"],
            "decoder_loss": ["timestep", "loss"],
            "reward_log": ["timestep", "position", "action", "idle time", "reward"]
        }
    elif cfg.version == "td3":
        fieldnames = {
            "log": ["timestep", "vehicle id", "position", "person list", "carbon emission", "cumulative distance"],
            "plog": ["timestep", "person id", "waiting time"],
            "solution_log": ["timestep", "vehicle id", "sequence of pickup locations", "sequence of drop-off locations",
                             "travel delay"],
            "service_rate_log": ["timestep", "service rate"],
            "encoder_loss": ["timestep", "train loss", "copy loss"],
            "decoder_loss": ["timestep", "critic1 loss", "critic2 loss", "actor loss"],
            "reward_log": ["timestep", "position", "action", "idle time", "reward"]
        }
    elif cfg.version in ("qmix_m", "qmix_a"):
        fieldnames = {
            "log": ["timestep", "vehicle id", "position", "person list", "carbon emission", "cumulative distance"],
            "plog": ["timestep", "person id", "waiting time"],
            "solution_log": ["timestep", "vehicle id", "sequence of pickup locations", "sequence of drop-off locations",
                             "travel delay"],
            "service_rate_log": ["timestep", "service rate"],
            "encoder_loss": ["timestep", "train loss", "copy loss"],
            "decoder_loss": ["timestep", "loss"],
            "reward_log": ["timestep", "previous states", "previous actions", "reward"]
        }


    logger = CSVLogger("./", date, fieldnames)

    if end is None:
        end = get_max_time()

    timestep = traci.simulation.getTime()
    start_time = round(timestep)

    # Initialize an empty solution for each vehicle
    solution: Solution = {v.vehicle_index: ([], []) for v in Vehicles}

    # Store vehicle counts per cluster for each second
    with open("data/label_to_edge_pos.pkl", "rb") as f:
        label_to_edge_pos = pickle.load(f)

    with open("data/allowed_edges_with_pos.pkl", "rb") as f:
        allowed_edges_with_pos = pickle.load(f)

    cluster_counts_by_second = {cid: deque(maxlen=cfg.interval) for cid in label_to_edge_pos}
    history_traffic_volume = deque(maxlen=cfg.n_hist + cfg.n_pred)

    running = True
    single_agent_transitions = {}
    qmix_transition = None
    while running:
        # Advance simulation by 1 second
        traci.simulationStep(timestep)

        if timestep % 100 == 0:
            print(f"--- Timestep: {timestep} ---")

        # Get the list of taxi vehicles
        sumo_fleet = traci.vehicle.getTaxiFleet(-1)
        taxi_set = set(sumo_fleet)

        for vid in traci.vehicle.getIDList():
            if vid in taxi_set:
                continue

            if traci.vehicle.getWaitingTime(vid) >= 300:
                try:
                    traci.vehicle.remove(vid)
                except traci.TraCIException:
                    pass

        for veh_id in sumo_fleet:
            # Insert a final stop if the vehicle's last stop is a drop-off
            logger.log("log", [timestep,
                               veh_id,
                               traci.vehicle.getRoadID(veh_id),
                               traci.vehicle.getPersonIDList(veh_id),
                               traci.vehicle.getCO2Emission(veh_id),
                               traci.vehicle.getDistance(veh_id)])
            next_stops = list(traci.vehicle.getNextStops(veh_id))

            if len(next_stops) == 0:
                try:
                    current_edge = traci.vehicle.getRoadID(veh_id).split("_")[0]
                    lane_id = traci.vehicle.getLaneID(veh_id)
                    if not lane_id:
                        lane_id = current_edge + "_0"
                    lane_length = traci.lane.getLength(lane_id)

                    traci.vehicle.changeTarget(veh_id, current_edge)
                    traci.vehicle.setStop(
                        veh_id,
                        current_edge,
                        pos=lane_length,
                        duration=cfg.idle_time,
                        flags=3
                    )
                except TraCIException:
                    pass

            passenger_list = traci.vehicle.getPersonIDList(veh_id)
            if len(passenger_list) == 0:
                # Empty
                if veh_id not in VehicleEmpty:
                    VehicleEmpty[veh_id] = timestep
                    EmptyCount[veh_id] = 0
            # Not empty
            if int(veh_id.split("_")[1]) in solution:
                VehicleEmpty[veh_id] = timestep
                EmptyCount[veh_id] = 0

        # Check for unassigned reservations
        reservations_not_assigned = traci.person.getTaxiReservations(3)
        for r_id in reservations_not_assigned:
            logger.log("plog", [timestep,
                                r_id.persons[0],
                                traci.person.getWaitingTime(r_id.persons[0])])

        # Calculate service rate
        if len(Serviced) + len(Rejected) > 0:
            logger.log("service_rate_log", [timestep,
                                            round(len(Serviced) / (len(Serviced) + len(Rejected)), 2)])

        for cluster_id, edges_pos in label_to_edge_pos.items():
            edges = list(edges_pos.keys())
            count_in_cluster = 0
            for edge_id in edges:
                if edge_id == '43203195' or edge_id == '511905952#0' or edge_id == '959767970':
                    continue
                try:
                    num_vehs_on_edge = traci.edge.getLastStepVehicleNumber(edge_id)
                    count_in_cluster += num_vehs_on_edge
                except:
                    continue

            cluster_counts_by_second[cluster_id].append(count_in_cluster)

        # If unassigned reservations exist, run dispatch
        if timestep % cfg.interval == 0:
            interval_index = (timestep - start_time + 1) // cfg.interval

            # Calculate average volume for each cluster in the last 'interval' seconds
            avg_volume = []
            for cluster_id in label_to_edge_pos:
                last_counts = list(cluster_counts_by_second[cluster_id])[-cfg.interval:]
                avg_count = sum(last_counts) / len(last_counts)
                avg_volume.append(avg_count)
            history_traffic_volume.append(avg_volume)

            unassigned_reservations = []
            for r in Reservations:
                in_solution = False
                for veh_index, (pickup_nodes, dropoff_nodes) in solution.items():
                    if (r.from_node in pickup_nodes) or (r.to_node in dropoff_nodes):
                        in_solution = True
                        break

                if not in_solution:
                    unassigned_reservations.append(r)

            # Optimization
            if reservations_not_assigned:
                data, solution = create_problem(
                    solution,
                    sumo_fleet,
                    cost_type,
                    waiting_time,
                    travel_delay,
                    int(end),
                    start_time,
                    timestep,
                    verbose
                )

                solution = dispatch(
                    start_time,
                    time_limit,
                    data,
                    solution,
                    verbose
                )

                sol_timestep = []
                sol_veh_id = []
                sol_pu_seq = []
                sol_do_seq = []
                sol_td = []
                for vid, (pu_list, do_list) in solution.items():
                    travel_delay_ = [-1 for _ in range(len(do_list))]
                    stops = pu_list + do_list

                    # ppu, pdo = list(), list()
                    ppu = [data.node_objects[pid].get_persons()[0] for pid in pu_list]
                    pdo = [data.node_objects[did].get_persons()[0] for did in do_list]
                    for idx, did in enumerate(do_list):
                        pid = data.node_objects[did].from_node

                        if pid in pu_list:
                            direct_travel_time = data.time_matrix[pid][did]
                            real_travel_time = 0
                            for seq in range(stops.index(pid), stops.index(did)):
                                real_travel_time += data.time_matrix[stops[seq]][stops[seq + 1]]
                            travel_delay_[idx] = real_travel_time - direct_travel_time

                    # Log solution result
                    logger.log("solution_log", [timestep, vid, ppu, pdo, travel_delay_])
                    sol_timestep.append(timestep)
                    sol_veh_id.append(vid)
                    sol_pu_seq.append(ppu)
                    sol_do_seq.append(pdo)
                    sol_td.append(travel_delay_)

                # Print out
                rows = []
                for t, vid, pu, do, td in zip(sol_timestep, sol_veh_id, sol_pu_seq, sol_do_seq, sol_td):
                    rows.append([t, vid, pu, do, td])
                print("[RIDE MATCHING RESULT]")
                print(tabulate(
                    rows,
                    headers=["time", "vehicle id", "pickup sequence", "drop-off sequence", "travel delay"],
                    tablefmt="fancy_grid",
                    showindex=False
                ))

            # Rebalancing
            if cfg.version in ("dqn", "td3", "qmix_m", "qmix_a"):
                if interval_index >= cfg.n_hist + cfg.n_pred:
                    x_train = z_score(
                        torch.tensor(np.array(history_traffic_volume), dtype=torch.float32, device=device),
                        cfg.mean,
                        cfg.std
                    )

                    x_train = x_train.unsqueeze(0).unsqueeze(-1)

                    y_hat, train_loss, copy_loss = encoder_trainer.train_step(torch.mul(x_train, 10))
                    y_hat = y_hat.squeeze(0).squeeze(-1)
                    y_hat = y_hat.detach().cpu().numpy()
                    logger.log("encoder_loss", [timestep, train_loss.item(), copy_loss.item()])

                    veh_pos = {}
                    veh_state = {}
                    global_state = y_hat[0]
                    for vid in VehicleEmpty:
                        # States
                        veh_index = int(vid.split("_")[1])
                        try:
                            x, y = traci.vehicle.getPosition(vid)
                            veh_pos[vid] = (x, y)
                            x_, y_ = map_normalization(x, y)
                            pos = np.array([[x_, y_]])
                            congestion_and_pos = np.concatenate((y_hat, pos), axis=-1)
                            global_state = np.append(global_state, x_)
                            global_state = np.append(global_state, y_)

                            dist_r = []
                            for r in unassigned_reservations:
                                if r in Reservations:
                                    px, py = traci.person.getPosition(r.get_persons()[0])
                                    px_norm, py_norm = map_normalization(px, py)
                                    dist = math.sqrt((x_ - px_norm) ** 2 + (y_ - py_norm) ** 2)
                                    dist_r.append((r, dist, (px_norm, py_norm)))

                            dist_r.sort(key=lambda tup: tup[1])
                            closest = dist_r[:cfg.c_passengers]

                            closest_pos = [[0, 0] for _ in range(cfg.c_passengers)]
                            for idx, (r, dist, (px_norm, py_norm)) in enumerate(closest):
                                closest_pos[idx][0] = px_norm
                                closest_pos[idx][1] = py_norm

                            closest_pos = np.reshape(np.array(closest_pos), (1, -1))
                            states = np.concatenate((congestion_and_pos, closest_pos), axis=-1)[0]

                            veh_state[vid] = states
                        except TraCIException:
                            pass

                        # Transitions
                        if cfg.version in ("dqn", "td3"):
                            reward = 0
                            if vid in single_agent_transitions:
                                print(f"[RECORD] Transitions for {vid}")
                                prev_state, prev_action = single_agent_transitions[vid]

                                idle_time = (timestep - VehicleEmpty[vid] + EmptyCount[vid] * cfg.idle_time) / 60
                                reward += -0.2 * (np.tanh(0.2 * (idle_time - (cfg.idle_time / 60))) + 1)
                                if len(Serviced) + len(Rejected) > 0:
                                    reward += 0.8 * len(Serviced) / (len(Serviced) + len(Rejected))

                                decoder_agent.store_transition(prev_state, prev_action, reward, veh_state[vid])

                                logger.log("reward_log", [timestep, veh_pos[vid], prev_action, idle_time, reward])
                                del single_agent_transitions[vid]

                            # Actions
                            acting = False
                            if (timestep - VehicleEmpty[vid]) >= cfg.idle_time:
                                if veh_index not in solution:
                                    agent_acting = False
                                    action_count = 0
                                    tried_actions = set()
                                    while True:
                                        if cfg.version == "dqn":
                                            action = decoder_agent.select_action(veh_state[vid], tried_actions)
                                            action_edge = find_nearest_edge_kdtree(veh_pos[vid][0],
                                                                                    veh_pos[vid][1],
                                                                                    label_to_edge_pos[action])
                                            tried_actions.add(action)
                                        elif cfg.version == "td3":
                                            action = decoder_agent.select_action(veh_state[vid])
                                            action_edge = find_nearest_edge_kdtree(action[0],
                                                                                   action[1],
                                                                                   allowed_edges_with_pos)

                                        try:
                                            n_next_stops = len(list(traci.vehicle.getNextStops(vid)))
                                            if n_next_stops == 0:
                                                traci.vehicle.changeTarget(vid, action_edge)
                                                traci.vehicle.setStop(
                                                    vid,
                                                    action_edge,
                                                    duration=cfg.idle_time,
                                                    flags=3
                                                )
                                                single_agent_transitions[vid] = (veh_state[vid], action)
                                                print(
                                                    f"[REBALANCING] {vid} moves to {action_edge} (action integer: {action})"
                                                )
                                                single_agent_transitions[vid] = (veh_state[vid], action)
                                                agent_acting = True
                                                acting = True

                                            elif n_next_stops == 1 and traci.vehicle.getNextStops(vid)[0][
                                                4] < -1 * cfg.idle_time:
                                                traci.vehicle.resume(vid)
                                                traci.vehicle.changeTarget(vid, action_edge)
                                                traci.vehicle.setStop(
                                                    vid,
                                                    action_edge,
                                                    duration=cfg.idle_time,
                                                    flags=3
                                                )
                                                single_agent_transitions[vid] = (veh_state[vid], action)
                                                print(
                                                    f"[REBALANCING] {vid} moves to {action_edge} (action integer: {action})"
                                                )
                                                agent_acting = True
                                                acting = True
                                            break

                                        except TraCIException:
                                            action_count += 1
                                            if action_count == 5:
                                                break
                                            continue

                                    if agent_acting:
                                        EmptyCount[vid] = 0
                                        VehicleEmpty[vid] = timestep
                                    else:
                                        EmptyCount[vid] += 1
                                        VehicleEmpty[vid] = timestep

                            if acting:
                                print(f"[UPDATE] {cfg.version.upper()}")
                                loss = decoder_agent.update()
                                print(f"{cfg.version.upper()} Loss: {loss}")
                                if loss:
                                    logger.log("decoder_loss", [timestep, loss.item()])

                    if cfg.version in ("qmix_m", "qmix_a"):
                        if qmix_transition:
                            print("[RECORD] Transitions for QMIX agents")
                            prev_states, prev_obs, agents, prev_actions = qmix_transition

                            reward = 0
                            for vid in veh_state:
                                idle_time = (timestep - VehicleEmpty[vid] + EmptyCount[vid] * cfg.idle_time) / 60
                                reward += -0.2 * (np.tanh(0.2 * (idle_time - (cfg.idle_time / 60))) + 1)
                                if len(Serviced) + len(Rejected) > 0:
                                    reward += 0.8 * len(Serviced) / (len(Serviced) + len(Rejected))

                            if veh_state:
                                veh_state_list = list(veh_state.values())
                                if len(veh_state_list) > cfg.n_agents:
                                    print(
                                        f"[WARNING] Active vehicles ({len(veh_state_list)}) exceeds n_agents ({cfg.n_agents}). Truncating.")
                                    veh_state_list = veh_state_list[:cfg.n_agents]
                                decoder_agent.store_transition(prev_states, prev_obs, prev_actions, reward,
                                                               global_state, veh_state_list)
                                logger.log("reward_log", [timestep, prev_states, prev_actions, reward])

                            qmix_transition = None

                        if veh_state:
                            acting = False

                            veh_state_items = list(veh_state.items())
                            if len(veh_state_items) > cfg.n_agents:
                                print(f"[INFO] Limiting active vehicles from {len(veh_state_items)} to {cfg.n_agents}")
                                veh_state_items = veh_state_items[:cfg.n_agents]
                                veh_state = dict(veh_state_items)

                            veh_state_list = [state for vid, state in veh_state_items]
                            veh_ids = [vid for vid, state in veh_state_items]

                            actions = decoder_agent.select_actions(veh_state_list)
                            qmix_transition = (global_state, veh_state_list, veh_ids, actions)

                            for idx, vid in enumerate(veh_state.keys()):
                                agent_acting = False
                                action = actions[idx]
                                state = veh_state[vid]
                                action_edge = find_nearest_edge_kdtree(veh_pos[vid][0], veh_pos[vid][1], label_to_edge_pos[action])

                                try:
                                    n_next_stops = len(list(traci.vehicle.getNextStops(vid)))
                                    if n_next_stops == 0:
                                        traci.vehicle.changeTarget(vid, action_edge)
                                        traci.vehicle.setStop(
                                            vid,
                                            action_edge,
                                            duration=cfg.idle_time,
                                            flags=3
                                        )
                                        single_agent_transitions[vid] = (veh_state[vid], action)
                                        print(
                                            f"[REBALANCING] {vid} moves to {action_edge} (action integer: {action})"
                                        )
                                        single_agent_transitions[vid] = (veh_state[vid], action)
                                        acting = True
                                        agent_acting = True

                                    elif n_next_stops == 1 and traci.vehicle.getNextStops(vid)[0][
                                        4] < -1 * cfg.idle_time:
                                        traci.vehicle.resume(vid)
                                        traci.vehicle.changeTarget(vid, action_edge)
                                        traci.vehicle.setStop(
                                            vid,
                                            action_edge,
                                            duration=cfg.idle_time,
                                            flags=3
                                        )
                                        single_agent_transitions[vid] = (veh_state[vid], action)
                                        print(
                                            f"[REBALANCING] {vid} moves to {action_edge} (action integer: {action})"
                                        )
                                        acting = True
                                        agent_acting = True

                                except TraCIException:
                                    pass

                                if agent_acting:
                                    EmptyCount[vid] = 0
                                    VehicleEmpty[vid] = timestep
                                else:
                                    EmptyCount[vid] += 1
                                    VehicleEmpty[vid] = timestep

                            if acting:
                                print(f"[UPDATE] {cfg.version.upper()}")
                                loss = decoder_agent.update()
                                if loss:
                                    print(f"{cfg.version.upper()} Loss: {loss}")
                                    if loss.item():
                                        logger.log("decoder_loss", [timestep, loss.item()])

        timestep += 1
        if timestep >= end:
            running = False

    logger.close()
    traci.close()
    sys.stdout.flush()
    return encoder_trainer, decoder_agent

if __name__ == "__main__":
    args = parse_args()

    # Overwrite configuration values
    cfg.sumo_config = args.sumo_config
    cfg.end = args.end
    cfg.idle_time = args.idle_time
    cfg.waiting_time = args.waiting_time
    cfg.travel_delay = args.travel_delay
    cfg.version = args.version
    cfg.n_agents = args.ncav
    cfg.ncav = args.ncav

    # Update osm.sumocfg.xml in terms of the number of CAVs
    update_route_file(config_path=cfg.sumo_config,
                      n_cav=cfg.ncav)

    # Set device
    if cfg.enable_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # Encoder: STGCN ------------------------------------------
    # Adjacency matrix loader
    adj = load_adj()

    # Calculate graph kernel
    L = scaled_laplacian(adj)

    # Alternative approximation: 1st approx - first_approx(W, n)
    Lk = torch.tensor(cheb_poly_approx(L, cfg.Ks, cfg.cluster_num),
                      dtype=torch.float32,
                      device=device)

    # Set encoder
    stgcn = STGCN(n_hist=cfg.n_hist,
                  Ks=cfg.Ks,
                  Kt=cfg.Kt,
                  blocks=cfg.blocks,
                  kernels=Lk,
                  dropout=0.0).to(device)

    # Define encoder loss function
    loss_fn = nn.MSELoss()

    # Set encoder optimizer
    optimizer = optim.Adam(stgcn.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    encoder_trainer = STGCNTrainer(model=stgcn,
                                   loss_fn=loss_fn,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   device=device)
    # --------------------------------------------------------

    # Decoder ------------------------------------------------
    if cfg.version == "dqn":
        cfg.obs_dim = cfg.cluster_num + 2 + cfg.c_passengers * 2
        decoder_agent = DQNAgent()
    elif cfg.version == "td3":
        cfg.obs_dim = cfg.cluster_num + 2 + cfg.c_passengers * 2
        decoder_agent = TD3Agent()
    elif cfg.version == "qmix_m":
        cfg.state_dim = cfg.cluster_num + cfg.n_agents * 2
        cfg.obs_dim = cfg.cluster_num + 2 + cfg.c_passengers * 2
        decoder_agent = QMIXMAgent()
    elif cfg.version == "qmix_a":
        cfg.state_dim = cfg.cluster_num + cfg.n_agents * 2
        cfg.obs_dim = cfg.cluster_num + 2 + cfg.c_passengers * 2
        decoder_agent = QMIXAAgent(n_agents=cfg.n_agents,
                                   obs_dim=cfg.obs_dim,
                                   state_dim=cfg.state_dim,
                                   act_dim=cfg.cluster_num)
    else:
        pass
    # --------------------------------------------------------

    today = datetime.now()
    formatted_date = today.strftime("%m%d%H%M")
    cfg.date = f"{formatted_date}_IT{cfg.idle_time}_WT{cfg.waiting_time}_TD{cfg.travel_delay}_{cfg.version}_NCAV{cfg.ncav}"
    os.makedirs(f"log/{cfg.date}", exist_ok=True)

    # Epoch
    best_service_rate = 0.0
    for epoch in range(cfg.epochs):
        print(f"--- Epoch #: {epoch+1}/{cfg.epochs} ---")

        if cfg.version in ("dqn", "qmix_m", "qmix_a"):
            decoder_agent.set_epoch(epoch, cfg.epochs)
            eps_info = decoder_agent.get_epsilon_info()
            print(f"[EPSILON] Strategy: {eps_info['strategy']}, Current ε: {eps_info['current_epsilon']:.4f}")

        Reservations = set()
        Vehicles = set()

        Pickup = set()
        Serviced = set()
        Rejected = set()

        PUOrder = []
        DOOrder = []
        Solution = {}
        CarbonEmission = {}
        LastDispatchedStops = {}

        GLOBAL_NODE_INDEX = {}
        GLOBAL_OBJECTS = {}

        epoch_dir = f"log/{cfg.date}/E{epoch+1}"
        os.makedirs(epoch_dir, exist_ok=True)

        # Decide which SUMO binary to use (GUI or non-GUI)
        if cfg.no_gui:
            sumoBinary = sumolib.checkBinary("sumo")
        else:
            sumoBinary = sumolib.checkBinary("sumo-gui")

        # Convert cost_type string to enum
        if cfg.cost_type == "distance":
            cost_type = CostType.DISTANCE
        elif cfg.cost_type == "time":
            cost_type = CostType.TIME
        else:
            raise ValueError(f"Wrong cost type {cfg.cost_type}")

        if cfg.waiting_time < 0:
            raise ValueError(f"Waiting time must be positive (got {cfg.waiting_time})")
        if cfg.travel_delay < 0:
            raise ValueError(f"Travel delay must be positive (got {cfg.travel_delay})")

        # Start SUMO simulation
        traci.start([sumoBinary,
                     "--no-warnings",
                     "-c", cfg.sumo_config,
                     "--seed", f"{epoch}",
                     "--max-depart-delay", "-1",
                     "--collision.action", "none"])

        # Run the main simulation loop
        encoder_trainer, decoder_agent = run(
            end=cfg.end,
            date=epoch_dir,
            time_limit=cfg.time_limit,
            cost_type=cost_type,
            waiting_time=cfg.waiting_time,
            travel_delay=cfg.travel_delay,
            encoder_trainer=encoder_trainer,
            decoder_agent=decoder_agent,
            device=device,
            verbose=cfg.verbose
        )

        # Calculate final service rate for this epoch
        final_service_rate = len(Serviced) / (len(Serviced) + len(Rejected)) if (len(Serviced) + len(
            Rejected)) > 0 else 0

        # Save best model if service rate improved
        if final_service_rate >= best_service_rate:
            best_service_rate = final_service_rate
            best_epoch = epoch

            # Save encoder model
            torch.save({
                'epoch': epoch,
                'model_state_dict': stgcn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'service_rate': best_service_rate,
            }, f"log/{cfg.date}/best_encoder_model_{epoch}.pth")

            # Save decoder model components
            if cfg.version == "dqn":
                torch.save({
                    'epoch': epoch,
                    'q_net_state_dict': decoder_agent.q_net.state_dict(),
                    'q_target_state_dict': decoder_agent.q_target.state_dict(),
                    'optimizer_state_dict': decoder_agent.optimizer.state_dict(),
                    'service_rate': best_service_rate,
                }, f"log/{cfg.date}/best_decoder_model_{epoch}.pth")
            elif cfg.version == "td3":
                torch.save({
                    'epoch': epoch,
                    'actor_state_dict': decoder_agent.actor.state_dict(),
                    'actor_target_state_dict': decoder_agent.actor_target.state_dict(),
                    'critic1_state_dict': decoder_agent.critic1.state_dict(),
                    'critic1_target_state_dict': decoder_agent.critic1_target.state_dict(),
                    'critic2_state_dict': decoder_agent.critic2.state_dict(),
                    'critic2_target_state_dict': decoder_agent.critic2_target.state_dict(),
                    'actor_optimizer_state_dict': decoder_agent.actor_optimizer.state_dict(),
                    'critic1_optimizer_state_dict': decoder_agent.critic1_optimizer.state_dict(),
                    'critic2_optimizer_state_dict': decoder_agent.critic2_optimizer.state_dict(),
                    'service_rate': best_service_rate,
                }, f"log/{cfg.date}/best_decoder_model_{epoch}.pth")
            elif cfg.version in ("qmix_m", "qmix_a"):
                torch.save({
                    'epoch': epoch,
                    'q_agent_nets_state_dict': decoder_agent.agent_nets.state_dict(),
                    'q_target_agent_nets_state_dict': decoder_agent.target_agent_nets.state_dict(),
                    'q_mixer_state_dict': decoder_agent.mixer.state_dict(),
                    'q_mixer_target_state_dict': decoder_agent.target_mixer.state_dict(),
                    'optimizer_state_dict': decoder_agent.optimizer.state_dict(),
                    'service_rate': best_service_rate,
                }, f"log/{cfg.date}/best_decoder_model_{epoch}.pth")

            print(f"New best model saved! Service rate: {best_service_rate:.4f} (Epoch {epoch})")
