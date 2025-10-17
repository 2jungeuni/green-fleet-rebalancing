<p align="center">
<h1 align="center">Green Fleet Rebalancing in AMoD Systems</h1>
</p>

This repository contains a SUMO simulation that models mixed traffic environments where autonomous robotaxis coexist with human-driven vehicles, aiming to maximize served passenger demand and reduce overall carbon emissions.

---

## Preliminary Setup
### Clone the repository

### Install SUMO
It is recommended to follow the installation instructions provided in the official [SUMO documentation](https://sumo.dlr.de/docs/Installing/index.html). 

### Set up environment
Create conda environment in the repository.
```bash
conda env create -f flow.yaml
conda activate flow
```

### Install dependencies
```bash
pip install traci sumolib gurobipy torch pybind11
```

### Generate a routing module
```bash
cd ./c_backend
c++ -O3 -Wall -shared -std=c++17 -fPIC `python -m pybind11 --includes` engine.cpp -o engine`python3-config --extension-suffix`
```
We implemented a heuristic based on Alonso-Mora's (2017, PNAS) vehicle-passenger matching algorithm to reduce the feasible solution space.
The heuristic is written in C++ and exposed as a Python module using Pybind. The optimization problem defined within this reduced feasible solution space is solved using Gurobipy in Python.

---

## Datasets
The SUMO environment settings are organized as below:  

├── env  
│ ├── env-ny  
│ │ ├── ny  
│ │ │ ├── osm.net.xml  
│ │ │ ├── osm.person.xml  
│ │ │ ├── osm.rou_X.xml  
│ │ │ ├── osm.flow_rev.rou.xml  
│ │ │ ├── vtypes.add.xml  
│ │ │ ├── osm.sumocfg.xml

* osm.net.xml: road network data (edge shapes, edge lengths, junctions, and more).
* osm.person.xml: records of each taxi request, including the request time, pickup location, and drop-off location.
* osm_rou_X.xml: initial position of CAVs (controlled autonomous vehicles).
* osm.flow_rev.rou.xml: start times and routes for HDVs (human-driven vehicles).
* vtypes.add.xml: vehicle type definitions.
* osm.sumocfg.xml: main SUMO simulation configuration file.
---

## Experiments
```bash
python3 run_ours.py -c <SUMO_CONFIGURATION_PATH> -v <ALGORITHM_NAME> -n <NUM_CAV> -e <END_TIMESTEP> -i <MAX_IDLE_TIME> -w <MAX_WAITING_TIME> -t <MAX_TRAVEL_DELAY> -it <NUM_ITERATION>
```
For example,
```bash
python3 run_ours.py -c "./env/env-ny/ny/osm.sumocfg.xml" -v "qmix_a" -n 5 -e 817200 -i 300 -w 300 -t 420 -it 5
```
It runs the ```QMIX-A``` algorithm for a three-hour simulation of five CAVs in New York City, enforcing a 5-minute idle limit, a 5-minute passenger wait limit, and a 7-minute maximum detour per request, repeated over five stochastic iterations.
