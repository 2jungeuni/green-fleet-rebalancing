import gurobipy as gp
from gurobipy import GRB
from framework.framework import Vehicle


def solve(existing_solution,
          feasible_solutions,
          data_vehicles,
          node_objects,
          time_matrix,
          dist_matrix):
    if existing_solution is None:
        existing_solution = {}

    def cal_solution_distance(veh_id, pu_list, do_list):
        for idx, veh in node_objects.items():
            if isinstance(veh, Vehicle):
                if veh.vehicle_index == veh_id:
                    v_loc = idx
                    break

        route = [v_loc] + list(pu_list) + list(do_list)

        total_dist = 0
        for i in range(len(route) - 1):
            n1 = route[i]
            n2 = route[i + 1]
            total_dist += dist_matrix[n1][n2]

        direct_dist = 0
        for j in do_list:
            pu = node_objects[j].from_node
            do = node_objects[j].to_node
            direct_dist += dist_matrix[pu][do]


        if direct_dist != 0:
            return total_dist / direct_dist
        else:
            return 30000

    for v_exist, (pu_exist, do_exist) in existing_solution.items():
        found = False
        for fs in feasible_solutions:
            if v_exist in fs:
                (pu_fs, do_fs) = fs[v_exist]
                if (pu_fs, do_fs) == (pu_exist, do_exist):
                    found = True
                    break

        if not found:
            feasible_solutions.append({v_exist: (pu_exist, do_exist)})

    model = gp.Model("ride-sharing")
    model.Params.LogToConsole = 0

    vehicles = set()
    nodes = set()
    num_dropoffs = 0

    solution_info = []

    for s_index, sol_dict in enumerate(feasible_solutions):
        (veh_id, (pu_list, do_list)) = list(sol_dict.items())[0]
        vehicles.add(veh_id)
        for n in pu_list: nodes.add(n)
        for n in do_list:
            nodes.add(n)
            num_dropoffs += 1

        solution_info.append((veh_id, pu_list, do_list))

    vehicles = sorted(list(vehicles))
    nodes = sorted(list(nodes))
    S = range(len(solution_info))

    vehicle_to_solution_indices = {v: [] for v in vehicles}
    for s_idx, (v, pu_list, do_list) in enumerate(solution_info):
        vehicle_to_solution_indices[v].append(s_idx)

    x = model.addVars(range(len(solution_info)), vtype=GRB.BINARY, name="x")

    y = model.addVars(nodes, vehicles, vtype=GRB.BINARY, name="y")

    M = model.addVars(vehicles, vtype=GRB.INTEGER, name="M")

    D = model.addVars(vehicles, vtype=GRB.CONTINUOUS, name="D")

    existing_assignments = {}
    for v, (pu_list, do_list) in existing_solution.items():
        for n in pu_list + do_list:
            existing_assignments[n] = v

    for n in nodes:
        if n in existing_assignments:
            v_assigned = existing_assignments[n]
            for v in vehicles:
                if v == v_assigned:
                    y[n, v].lb = 1
                    y[n, v].ub = 1
                else:
                    y[n, v].lb = 0
                    y[n, v].ub = 0

    for v in vehicles:
        model.addConstr(
            gp.quicksum(x[s_idx] for s_idx in vehicle_to_solution_indices[v]) <= 1,
            name=f"one_solution_per_vehicle_v{v}"
        )

    for n in nodes:
        model.addConstr(
            gp.quicksum(y[n, v] for v in vehicles) == 1,
            name=f"assign_node_{n}"
        )

    solution_distance = {}
    for s_idx, (v_s, pu_list, do_list) in enumerate(solution_info):
        for n in pu_list:
            model.addConstr(
                x[s_idx] <= y[n, v_s],
                name=f"pu_s{s_idx}_n{n}"
            )
        for n in do_list:
            model.addConstr(
                x[s_idx] <= y[n, v_s],
                name=f"do_s{s_idx}_n{n}"
            )
        solution_distance[s_idx] = cal_solution_distance(v_s, pu_list, do_list)

    for v in vehicles:
        model.addConstr(
            M[v] == gp.quicksum(len(solution_info[s_idx][2]) * x[s_idx]
                                for s_idx in vehicle_to_solution_indices[v]),
            name=f"num_dropoffs_v{v}"
        )
        model.addConstr(
            D[v] == gp.quicksum(solution_distance[s_idx] * x[s_idx]
                                for s_idx in vehicle_to_solution_indices[v]),
            name=f"distance_v{v}"
        )

    model.ModelSense = GRB.MAXIMIZE

    if num_dropoffs > 0:
        model.setObjectiveN(
            gp.quicksum(M[v] for v in vehicles) / num_dropoffs,
            index=0,
            priority=2,
        )
    else:
        model.setObjectiveN(0, index=0, priority=2,)

    model.setObjectiveN(
        gp.quicksum(-1 * D[v] for v in vehicles),
        index=1,
        priority=1
    )

    model.optimize()

    solutions = {}
    if model.status == GRB.OPTIMAL:
        for s_idx in S:
            if x[s_idx].X > 0.5:
                (veh_id, pu_list, do_list) = solution_info[s_idx]
                solutions[veh_id] = (pu_list, do_list)
        return solutions, model.ObjVal
    else:
        return existing_solution, 0