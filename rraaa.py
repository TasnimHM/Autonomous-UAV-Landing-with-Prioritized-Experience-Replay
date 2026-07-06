#! /usr/bin/python3

import argparse
import atexit
import signal
import sys
import time

from utils import constants
from utils.config import parse_hierarchical_config, write_flattened_config, read_shared_tmp_file
from utils.docker import ContainerManager
from utils.logging import set_logger, log


def get_args():
    parser = argparse.ArgumentParser(description="RRAAA Simulator for Autonomous Air Taxis.")
    parser.add_argument("config", type=str, help="Path to the config file")
    return parser.parse_args()


class Test:

    def __init__(self, config_path):
        self.config = parse_hierarchical_config(config_path)
        write_flattened_config(constants.merged_config_path, self.config)
        set_logger(self.config['loglevel'])

        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown_handler)
        signal.signal(signal.SIGTERM, self.shutdown_handler)

        self.containermanager = ContainerManager(self.config, constants.compose_file)

    def shutdown(self, signal_number=None):
        """Gracefully shuts down all containers and processes."""
        if hasattr(self, 'containermanager') and self.containermanager:
            self.containermanager.terminate_all()
        if signal_number is None:
            log.info("Terminated.")
        else:
            log.info(f"Terminated with signal {signal_number}")
        sys.exit(0)

    def shutdown_handler(self, signal_number, frame):
        self.shutdown(signal_number)

    def simulation_ended(self, filename):
        try:
            return read_shared_tmp_file(filename) == 'True'
        except FileNotFoundError:
            return False

    def reset(self):
        """Resets containers and simulation status for the next run."""
        write_flattened_config(constants.merged_config_path, self.config)
        try:
            with open(constants.simulation_status_file, 'w') as file:
                file.write('False')
        except FileNotFoundError:
            log.warning(f"Status file {constants.simulation_status_file} not found. Creating a new one.")
            with open(constants.simulation_status_file, 'w') as file:
                file.write('False')
        self.containermanager.stop_all()
        self.containermanager.start_all()
        self.containermanager.run_all()

    def run_once(self):
        """Runs a single simulation iteration."""
        try:
            self.containermanager.start_all()
            self.containermanager.build_all_workspaces()
            self.containermanager.run_all()
            #self.containermanager.wait_for_all()

            while not self.simulation_ended(constants.simulation_status_file):

                time.sleep(1)  # Poll every second
                

            log.info("Simulation completed successfully.")

        except Exception as e:
            pass 
        #     log.error(f"An error occurred during run_once: {e}")
        # finally:
        #     log.info("Attempting to terminate all containers.")
        #     try:
        #         #self.containermanager.terminate_all()
        #     except Exception as e:
        #         log.error(f"Error during termination: {e}")
        #         self.containermanager.force_stop_all_containers()

    def run_iterations(self, iter_count):
        """Runs multiple simulation iterations."""
        self.containermanager.start_all()
        self.containermanager.build_all_workspaces()

        for iteration in range(iter_count):
            log.info(f"Starting Iteration {iteration + 1}")
            self.containermanager.run_all()

            while not self.simulation_ended(constants.landing_target_reached_file):
                time.sleep(1)

            if iteration < iter_count - 1:
                self.reset()
            else:
                self.containermanager.stop_all()

    def run(self):
        """Determines the simulation mode and runs accordingly."""
        mode = self.config['simulation_mode']
        if mode == 'simple':
            self.run_once()
        elif isinstance(mode, int):
            self.run_iterations(mode)
        else:
            log.error(f"Unknown simulation mode {mode}")
            sys.exit(-1)


if __name__ == "__main__":
    args = get_args()
    test = Test(args.config)
    test.run()
