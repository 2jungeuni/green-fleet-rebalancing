# built-in
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union, Set

import traci
import traci._person
import traci._simulation

SPEED_DEFAULT = 20


class CostType(Enum):
    DISTANCE = 1
    TIME = 2

"""
Key: (edge_from, edge_to, vehicleType)
Value: Stage from traci.simulation.findRoute(...)
"""
ROUTE_CACHE: Dict[Tuple[str, str, str], traci._simulation.Stage] = {}

def _find_route_cached(edge_from: str, edge_to: str, vehicle_type: str) -> traci._simulation.Stage:
    if (edge_from, edge_to) in ROUTE_CACHE:
        return ROUTE_CACHE[(edge_from, edge_to, vehicle_type)]

    stage = traci.simulation.findRoute(edge_from, edge_to, vType=vehicle_type)
    ROUTE_CACHE[(edge_from, edge_to, vehicle_type)] = stage
    return stage


@dataclass
class Node:
    """
    Connects an object of the routing problem with a nodeID.
    """
    class NodeType(Enum):
        FROM_EDGE = 1
        TO_EDGE = 2
        VEHICLE = 3
        DEPOT = 4

    node_type: NodeType


@dataclass
class Vehicle:
    """
    Represents a vehicle / route for the routing problem.
    """
    id_vehicle: str
    vehicle_index: int
    num_passengers: int = 0
    start_time: int = None
    start_node: int = None
    end_node: int = None

    def __eq__(self, other):
        if not isinstance(other, Vehicle):
            return False
        return self.id_vehicle == other.id_vehicle

    def __hash__(self):
        return hash(self.id_vehicle)

    def get_person_capacity(self) -> int:
        return traci.vehicle.getPersonCapacity(self.id_vehicle)

    def get_type_ID(self) -> str:
        return traci.vehicle.getTypeID(self.id_vehicle)

    def get_edge(self) -> str:
        return traci.vehicle.getRoadID(self.id_vehicle)

    def get_person_id_list(self) -> List[str]:
        return traci.vehicle.getPersonIDList(self.id_vehicle)


@dataclass
class Reservation:
    """
    Represents a request for a transportation.
    """
    reservation: traci.person.Reservation
    from_node: int = Node
    to_node: int = Node

    # for detour ratio
    direct_route_cost: int = None
    current_route_cost: int = None
    vehicle: Vehicle = None

    # four time window
    pickup_earliest: float = None
    dropoff_latest: float = None

    def __eq__(self, other):
        if not isinstance(other, Reservation):
            return False
        return self.reservation.id == other.reservation.id

    def __hash__(self):
        return hash(self.reservation.id)

    def is_new(self) -> bool:
        if self.reservation.state == 1 or self.reservation.state == 2:
            return True
        else:
            return False

    def is_picked_up(self) -> bool:
        return self.reservation.state == 8

    def is_done(self) -> bool:
        return self.reservation not in traci.person.getTaxiReservations(0)

    def is_from_node(self, node: int) -> bool:
        return (not self.is_picked_up() and self.from_node == node)

    def is_to_node(self, node: int) -> bool:
        return self.to_node == node

    def get_from_edge(self) -> str:
        return self.reservation.fromEdge

    def get_to_edge(self) -> str:
        return self.reservation.toEdge

    def get_id(self) -> str:
        return self.reservation.id

    def get_persons(self) -> List[str]:
        return self.reservation.persons

    def get_earliest_pickup(self) -> int:
        person_id = self.get_persons()[0]
        self.pickup_earliest = (
                traci.person.getParameter(person_id, "pickup_earliest")
                or traci.person.getParameter(person_id, "earliestPickupTime")
                or self.pickup_earliest
        )

        if self.pickup_earliest:
            self.pickup_earliest = round(float(self.pickup_earliest))
        return self.pickup_earliest

    def get_dropoff_latest(self) -> int:
        person_id = self.get_persons()[0]
        self.dropoff_latest = (
            traci.person.getParameter(person_id, "dropoff_latest")
            or traci.person.getParameter(person_id, "latestDropoffTime")
            or self.dropoff_latest
        )

        if self.dropoff_latest:
            self.dropoff_latest = round(float(self.dropoff_latest))
        return self.dropoff_latest

    def get_direct_route_cost(self,
                              type_vehicle: str,
                              cost_matrix: Union[List[List[int]], None] = None,
                              cost_type: CostType = CostType.TIME):
        # if the direct route cost is already calculated,
        if self.direct_route_cost:
            return

        # if the reservation is assigned to a vehicle,
        if not self.is_picked_up():
            self.direct_route_cost = cost_matrix[self.from_node][self.to_node]

        # if it doesn't have any schedule,
        else:
            route: traci._simulation.Stage = traci.simulation.findRoute(
                self.get_from_edge(),
                self.get_to_edge(),
                vType=type_vehicle
            )
            if cost_type == CostType.TIME:
                self.direct_route_cost = round(route.travelTime)
                return self.direct_route_cost
            elif cost_type == CostType.DISTANCE:
                self.direct_route_cost = round(route.length)
                return self.direct_route_cost
            else:
                raise ValueError(f"Cannot set given cost ({cost_type}).")

    def update_direct_route_cost(self,
                                 type_vehicle: str,
                                 cost_matrix: Union[List[List[int]], None] = None,
                                 cost_type: CostType = CostType.TIME):
        # if the direct route cost is already calculated.
        if self.direct_route_cost:
            return

        # if the reservation is assigned to a vehicle,
        if not self.is_picked_up():
            self.direct_route_cost = cost_matrix[self.from_node][self.to_node]

        # if it doesn't have any schedule,
        else:
            route: traci._simulation.Stage = traci.simulation.findRoute(
                self.get_from_edge(),
                self.get_to_edge(),
                vType=type_vehicle
            )
            if cost_type == CostType.TIME:
                self.direct_route_cost = round(route.travelTime)
            elif cost_type == CostType.DISTANCE:
                self.direct_route_cost = round(route.length)
            else:
                raise ValueError(f"Cannot set given cost ({cost_type}).")

    def update_current_route_cost(
            self,
            cost_type: CostType = CostType.TIME
    ):
        person_id = self.reservation.persons[0]
        stage: traci._simulation.Stage = traci.person.getStage(person_id, 0)

        if cost_type == CostType.DISTANCE:
            self.current_route_cost = round(stage.length)
        elif cost_type == CostType.TIME:
            self.current_route_cost = round(stage.travelTime)
        else:
            raise ValueError(f"Cannot set given cost({cost_type}).")


NodeObject = Union[str, Vehicle, Reservation]


@dataclass
class Framework:
    """
    Data model class used by constraints of the OR-tools lib.
    """
    # node ID of the depot
    depot: int
    cost_matrix: List[List[int]]
    time_matrix: List[List[int]]
    dist_matrix: List[List[int]]
    num_vehicles: int
    starts: List[int]
    ends: List[int]
    waiting_time: int
    travel_delay: int
    time_windows: Dict[int, Tuple[int, int]]
    max_time: int
    node_objects: Dict[int, NodeObject]
    reservations: Set[Reservation]
    vehicles: Set[Vehicle]
    cost_type: CostType

def reject_late_reservations(
        data_reservations: Set[Reservation],
        vehicles: Set[Vehicle],
        waiting_time: int,
        timestep: float
) -> Tuple[Set[Reservation], List[Reservation]]:
    """
    Rejects reservations that are not assigned to a vehicle and cannot be served by time.
    Returns a cleared list and a list of the remove reservations.
    """
    rejected_reservations = []

    all_res = data_reservations.copy()

    for res in data_reservations:
        if not res.vehicle and res.reservation.reservationTime + waiting_time <= timestep:
            all_res.remove(res)
            rejected_reservations.append(res)

    return all_res, rejected_reservations

def get_edge_of_node_object(
        node_object: NodeObject,
        node: int
) -> Union[str, None]:
    if isinstance(node_object, Vehicle):
        return node_object.get_edge()
    if isinstance(node_object, Reservation):
        if node_object.is_from_node(node):
            return node_object.get_from_edge()
        if node_object.is_to_node(node):
            return node_object.get_to_edge()
    return None

def get_time_of_node_object(
        node_object: NodeObject,
        node: int
) -> Union[int, None]:
    if isinstance(node_object, Vehicle):
        return node_object.start_time
    if isinstance(node_object, Reservation):
        return round(node_object.pickup_earliest)
    return None

def get_cost_matrix(
        node_objects: Dict[int, NodeObject],
        start_time: int,
        cost_type: CostType
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Get cost matrix between edges.
    Index in cost matrix is the same as the node index of the constraint solver.
    """
    if not node_objects:
        return [], []

    max_idx=  max(node_objects.keys())
    size = max_idx + 1

    time_matrix = [[30000 for _ in range(size)] for __ in range(size)]
    cost_matrix = [[30000 for _ in range(size)] for __ in range(size)]
    dist_matrix = [[30000 for _ in range(size)] for __ in range(size)]

    vehicle_type = None
    any_vehicle = next((obj for obj in node_objects.values() if isinstance(obj, Vehicle)), None)
    vehicle_type = any_vehicle.get_type_ID()

    # boarding / pickup / dropoff duration
    boardingDuration_param = traci.vehicletype.getBoardingDuration(vehicle_type)
    boardingDuration = 0 if boardingDuration_param == "" else round(float(boardingDuration_param))

    # pickup nd dropoff duration of first vehicle is used for all vehicles
    pickUpDuration_param = traci.vehicle.getParameter(any_vehicle.id_vehicle, "device.taxi.pickUpDuration")
    pickUpDuration = 0 if pickUpDuration_param == "" else round(float(pickUpDuration_param))
    dropOffDuration_param = traci.vehicle.getParameter(any_vehicle.id_vehicle, "device.taxi.dropOffDuration")
    dropOffDuration = 0 if dropOffDuration_param == "" else round(float(dropOffDuration_param))

    for i, from_obj in node_objects.items():
        edge_from = get_edge_of_node_object(from_obj, i)
        time_from = get_time_of_node_object(from_obj, i)

        time_matrix[i][i] = 0
        cost_matrix[i][i] = 0

        for j, to_obj in node_objects.items():
            if i == j:
                continue

            edge_to = get_edge_of_node_object(to_obj, j)
            time_to = get_time_of_node_object(to_obj, j)

            # Depot
            if isinstance(from_obj, str) and from_obj == "depot":
                if time_to >= start_time:
                    travel_t = time_to - start_time
                else:
                    travel_t = 0
                time_matrix[i][j] = travel_t
                cost_matrix[i][j] = travel_t if cost_type == CostType.TIME else 0
                dist_matrix[i][j] = 0
                continue

            if isinstance(to_obj, str) and to_obj == "depot":
                time_matrix[i][j] = 0
                cost_matrix[i][j] = 0
                dist_matrix[i][j] = 0
                continue

            if edge_from is None or edge_to is None:
                continue

            if edge_from == '' or edge_to == '':
                continue

            route = _find_route_cached(edge_from, edge_to, vehicle_type) if vehicle_type else None
            if not route or len(route.edges) == 0:
                continue

            travel_time = round(route.travelTime)
            travel_dist = round(route.length)

            # Pickup
            if isinstance(from_obj, Reservation) and from_obj.is_from_node(i):
                travel_time += pickUpDuration
                travel_time += boardingDuration

            # Dropoff
            if isinstance(to_obj, Reservation) and to_obj.is_to_node(j):
                travel_time += dropOffDuration

            time_matrix[i][j] = travel_time
            dist_matrix[i][j] = travel_dist
            if cost_type == CostType.TIME:
                cost_matrix[i][j] = travel_time
            else:
                cost_matrix[i][j] = travel_dist

    return time_matrix, dist_matrix, cost_matrix

def get_time_window_of_node_object(
        node_object: NodeObject,
        node: int,
        end: int,
        start_time: int,
        waiting_time: int
) -> Tuple[int, int]:
    """
    Returns a pair with earliest and latest service time.
    """
    time_window = None
    if isinstance(node_object, str) and node_object == "depot":
        time_window = (0, round(end) - start_time)

    elif isinstance(node_object, Vehicle):
        time_window = (node_object.start_time - start_time, round(end) - start_time)

    elif isinstance(node_object, Reservation):
        if node_object.is_from_node(node):
            time_window = (
                node_object.pickup_earliest - start_time,
                node_object.pickup_earliest - start_time + waiting_time
            )

        if node_object.is_to_node(node):
            time_window = (
                node_object.pickup_earliest - start_time,
                round(end) - start_time
            )

    else:
        raise ValueError(f"Cannot set time window for node {node}.")

    return time_window


