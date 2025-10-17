#ifndef HELPER_H
#define HELPER_H

#include <map>
#include <tuple>
#include <vector>
#include <string>
#include <utility>
#include <iostream>

using StartEnd = std::pair<int, int>;
using TimeMatrix = std::vector<std::vector<int>>;
using TimeWindow = std::unordered_map<int, std::pair<int, int>>;
using Bundle = std::tuple<std::vector<int>, std::vector<int>>;
using FeasibleSolutions = std::vector<std::map<int, Bundle>>;

#define sz(v) ((int)(v).size())
#define REP0(i, n) for(int i = 0; i < n; i++)
#define REP1(i, n) for(int i = 1; i <= n; i++)
#define REP(i, a, b) for (int i = a; i <= b; i++)

std::ostream& printVector(std::ostream& os, const std::vector<int>& v) {
  os << "[";
  for (size_t i = 0; i < v.size(); ++i) {
    os << v[i];
    if (i + 1 < v.size()) {
      os << ", ";
    }
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const FeasibleSolutions& fs) {
  os << "Feasible solutions (size: " << fs.size() << ")\n";
  for (size_t i = 0; i < fs.size(); ++i) {
    os << "\t- Candidate" << i << ":\n";
    for (const auto& kv: fs[i]) {
      os << "\t\tVehicle ID: " << kv.first << " => ";

      const auto& bundle = kv.second;
      const auto& vec1 = std::get<0>(bundle);
      const auto& vec2 = std::get<1>(bundle);

      os << "(";
      printVector(os, vec1) << ", ";
      printVector(os, vec2) << ")";
      os << "\n";
    }
  }
  return os;
}

void printOrders(const std::vector<int>& pu_order, const std::vector<int>& do_order) {
    std::cout << "PU Orders: ";
    for (int value: pu_order) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    std::cout << "DO Orders: ";
    for (int value: do_order) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

#endif //HELPER_H