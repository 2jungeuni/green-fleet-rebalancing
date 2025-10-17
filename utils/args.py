import argparse
from tabulate import tabulate

from config import cfg

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AMoD simulation with SUMO and TraCI"
    )

    parser.add_argument(
        "--sumo-config",
        "-c",
        type=str,
        default=cfg.sumo_config,
        help="Path to the SUMO configuration file (.sumocfg or .sumocfg.xml)"
    )

    parser.add_argument(
        "--version",
        "-v",
        type=str,
        choices=["default", "rl", "ha", "dqn", "td3", "qmix_m", "qmix_a"],
        default="default",
        help="Simulation version (default, rl, ha, dqn, td3, qmix_m, or qmix_a)"
    )

    parser.add_argument(
        "--ncav",
        "-n",
        type=int,
        default=3,
        help="Number of Controlled Autonomous Vehicles (CAVs)"
    )

    parser.add_argument(
        "--end",
        "-e",
        type=int,
        default=cfg.end,
        help="Simulation end time (seconds)"
    )

    parser.add_argument(
        "--idle-time",
        "-i",
        type=int,
        default=cfg.idle_time,
        help="Maximum time a vehicle can remain idle (seconds)"
    )

    parser.add_argument(
        "--waiting-time",
        "-w",
        type=int,
        default=cfg.waiting_time,
        help="Maximum waiting time for reservation (seconds)"
    )

    parser.add_argument(
        "--travel-delay",
        "-t",
        type=int,
        default=cfg.travel_delay,
        help="Allowed maximum travel delay (seconds)"
    )

    parser.add_argument(
        "--iteration",
        "-it",
        type=int,
        default=cfg.iteration,
        help="Number of iterations"
    )

    args = parser.parse_args()
    table = tabulate(
        vars(args).items(),
        headers=["Argument", "Value"],
        tablefmt="fancy_grid",
        showindex=False
    )
    print(table)
    return args