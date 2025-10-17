#ifndef RUN_H
#define RUN_H

#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "helper.h"
#include "constraints.h"

FeasibleSolutions run(
        std::vector<StartEnd>& res_info,
        std::map<int, StartEnd>& veh_info,
        TimeMatrix& time_matrix,
        TimeWindow& time_window,
        std::map<int, Bundle>& solution,
        const int waiting_time,
        const int travel_delay
) {
    std::cout << "[INFO] Start c++ engine with new requests " << sz(res_info)  << std::endl;

    std::unordered_set<int> used_nodes;
    for (const auto& [veh_id, bundle] : solution) {
        const auto& pu = std::get<0>(bundle);
        const auto& do_ = std::get<1>(bundle);

        used_nodes.insert(pu.begin(), pu.end());
        used_nodes.insert(do_.begin(), do_.end());
    }

    FeasibleSolutions all_feasible_solutions;

    for (size_t r = 0; r < res_info.size(); r++) {
        const StartEnd& req = res_info[r];

        int pickup_node = req.first;
        int dropoff_node = req.second;

        if (used_nodes.count(pickup_node) || used_nodes.count(dropoff_node)) {
            std::cout << "[SKIP] Request (PU=" << pickup_node << ", DO=" << dropoff_node << ") already exists in solution.\n";
            continue;
        }

        for (auto& veh: veh_info) {
            int veh_id = veh.first;

            FeasibleSolutions feasible_sol = TryInsertReservation(
                veh_id,
                req,
                veh_info,
                time_matrix,
                time_window,
                solution,
                waiting_time,
                travel_delay
            );

            all_feasible_solutions.insert(
                all_feasible_solutions.end(),
                feasible_sol.begin(),
                feasible_sol.end()
            );
        }
    }

    return all_feasible_solutions;
}

#endif //RUN_H