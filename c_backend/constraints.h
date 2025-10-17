#ifndef CONSTRAINTS_H
#define CONSTRAINTS_H

#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <utility>
#include <iostream>
#include <unordered_set>

#include "helper.h"
#include "constraints.h"

bool CheckFeasibility(
    const std::vector<int>& pu_order,
    const std::vector<int>& do_order,
    int pickup_loc,
    int dropoff_loc,
    int vehicle_id,
    const std::map<int, StartEnd>& veh_info,
    const TimeMatrix& time_matrix,
    const TimeWindow& time_window,
    const int waiting_time,
    const int travel_delay
) {
  int current_loc = veh_info.at(vehicle_id).first;
  int current_time = time_window.at(current_loc).first;
  int on_board = 0;

  int direct_travel_time = 0;
  if (pickup_loc != -1) {
    direct_travel_time = time_matrix[pickup_loc][dropoff_loc];
  }
  int pickup_time = -1;
  int dropoff_time = -1;

  std::vector<std::pair<int, char>> route;
  for (auto loc : pu_order) {
    route.push_back({loc, 'P'});
  }
  for (auto loc : do_order) {
    route.push_back({loc, 'D'});
  }

  bool seenDropOff = false;
  for (auto& [loc, tp]: route) {
    if (tp == 'D') {
      seenDropOff = true;
    } else if (tp == 'P' && seenDropOff) {
      return false;
    }
  }

  for (size_t k = 0; k < route.size(); k++) {
    auto [loc, tp] = route[k];
    auto [earliest, latest] = time_window.at(loc);

    int travel = time_matrix[current_loc][loc];
    current_time += travel;
    current_loc = loc;

    if (tp == 'P') {
        if (loc == pickup_loc && pickup_loc != -1) {
            pickup_time = current_time;
            if (pickup_time < earliest) return false;
            if (pickup_time > earliest + waiting_time) {
              return false;
            }
        }
        on_board++;
        if (on_board > 3) return false;
    }
    else if (tp == 'D') {
      if (loc == dropoff_loc) {
        dropoff_time = current_time;

        if (pickup_loc != -1) {
          if (dropoff_time < pickup_time) {
            return false;
          }

          if (dropoff_time - pickup_time > direct_travel_time + travel_delay) {
            return false;
          }
        }
      }

      on_board--;
      if (on_board < 0) {
        return false;
      }
    }

    if (current_time < earliest) {
      return false;
    }
    if (current_time > latest) {
      return false;
    }
  }

  return true;
}

FeasibleSolutions TryInsertReservation(
    int vehicle_id,
    const StartEnd& reservation,
    const std::map<int, StartEnd>& veh_info,
    const TimeMatrix& time_matrix,
    const TimeWindow& time_window,
    std::map<int, Bundle>& solution,
    const int waiting_time,
    const int travel_delay
) {
  FeasibleSolutions feasible_sol;

  auto& bundle = solution[vehicle_id];
  auto& pu_order = std::get<0>(bundle);
  auto& do_order = std::get<1>(bundle);

  int pickup_loc = reservation.first;
  int dropoff_loc = reservation.second;

  if (pickup_loc == -1) {
    for (int j = 0; j <= sz(do_order); j++) {
     	do_order.insert(do_order.begin() + j, dropoff_loc);

		bool feasible = CheckFeasibility(
            pu_order,
            do_order,
            pickup_loc,
            dropoff_loc,
            vehicle_id,
            veh_info,
            time_matrix,
            time_window,
            waiting_time,
            travel_delay
		);

        if (feasible) {
          std::map<int, Bundle> sol_candidate;

          sol_candidate[vehicle_id] = std::make_tuple(pu_order, do_order);
          feasible_sol.push_back(sol_candidate);
        }

        do_order.erase(do_order.begin() + j);
    }
  } else {
    for (int i = 0; i <= sz(pu_order); i++) {
      for (int j = 0; j <= sz(do_order); j++) {
        pu_order.insert(pu_order.begin() + i, pickup_loc);
        do_order.insert(do_order.begin() + j, dropoff_loc);

        bool feasible = CheckFeasibility(
            pu_order,
            do_order,
            pickup_loc,
            dropoff_loc,
            vehicle_id,
            veh_info,
            time_matrix,
            time_window,
            waiting_time,
            travel_delay
        );

        if (feasible) {
          std::map<int, Bundle> sol_candidate;
          sol_candidate[vehicle_id] = std::make_tuple(pu_order, do_order);
          feasible_sol.push_back(sol_candidate);
        }

        pu_order.erase(pu_order.begin() + i);
      	do_order.erase(do_order.begin() + j);
      }
    }
  }

  return feasible_sol;
}

#endif