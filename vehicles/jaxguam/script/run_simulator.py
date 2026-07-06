#!/usr/bin/env python3
import os
import time
import yaml
import rospy
import numpy as np
import subprocess
import argparse
from threading import Lock
from geometry_msgs.msg import PoseStamped, Vector3
from tf.transformations import euler_from_quaternion
from utils import constants
from loguru import logger
from plot_results import create_run_folder, save_results_to_csv, plot_initial_pos, plot_final_pos, save_summary_to_csv_and_metadata  # Import from plot_results


config_path = constants.merged_config_path
temp_config_path = '/tmp/tmp_config.yml'

ideal_x, ideal_y, ideal_z = -80, 75, 35
range_offset = [40, 40, 120]

x_min, x_max = ideal_x - range_offset[0], ideal_x + range_offset[0]
y_min, y_max = ideal_y - range_offset[1], ideal_y + range_offset[1]
z_min, z_max = ideal_z + 20, ideal_z + range_offset[2]

initial_positions = []
final_positions = []
landing_times = []
landing_results = []
final_euler_angles = []


class SimulationMonitor:
    def __init__(self):
        self.lock = Lock()
        self.z_value_below_threshold = False
        self.zero_velocity_duration = 0
        self.velocity_zero_start_time = None
        self.current_pose = None
        self.final_angles = None

        rospy.Subscriber('/jaxguam/pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/controller_node/vel_cmd', Vector3, self.velocity_callback)

    def pose_callback(self, msg):
        """Callback to check z-value and save current position."""
        with self.lock:
            self.current_pose = msg.pose.position
            if msg.pose.position.z < ideal_z:
                self.z_value_below_threshold = True
                quaternion = (
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w
                )
                self.final_angles = euler_from_quaternion(quaternion)

    def velocity_callback(self, msg):
        """Callback to check velocity."""
        with self.lock:
            if msg.x == 0 and msg.y == 0 and msg.z == 0:
                if self.velocity_zero_start_time is None:
                    self.velocity_zero_start_time = time.time()
                self.zero_velocity_duration = time.time() - self.velocity_zero_start_time
            else:
                self.velocity_zero_start_time = None
                self.zero_velocity_duration = 0

    def reset(self):
        """Reset state for the next iteration."""
        with self.lock:
            self.z_value_below_threshold = False
            self.zero_velocity_duration = 0
            self.velocity_zero_start_time = None
            self.current_pose = None
            self.final_angles = None


def run_simulation(monitor, init_x, init_y, init_z, timeout=150):
    """Run a single simulation with provided initial conditions and timeout."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['ego_vehicle']['location']['x'] = init_x
    config['ego_vehicle']['location']['y'] = init_y
    config['ego_vehicle']['location']['z'] = init_z

    with open(temp_config_path, 'w') as file:
        yaml.safe_dump(config, file)

    # Initialize simulation status file to 'False'
    with open(constants.simulation_status_file, 'w') as status_file:
        status_file.write('False')

    process = subprocess.Popen(
        ["rosrun", "jaxguam", "land_vtol.py", "--config", temp_config_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    start_time = time.time()

    try:
        while process.poll() is None:
            elapsed_time = time.time() - start_time

            if monitor.z_value_below_threshold:
                logger.warning(f"Iteration stopped: Z-value dropped below {ideal_z}m")
                break

            if monitor.zero_velocity_duration >= 20:
                logger.warning("Iteration stopped: Velocities zero for 20 seconds.")
                break

            if elapsed_time > timeout:
                logger.warning(f"Iteration stopped: Timeout reached after {timeout}s.")
                break

            rate.sleep()

    except KeyboardInterrupt:
        logger.error("Keyboard interrupt detected. Terminating process...")

    finally:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
            logger.info(f"Subprocess output: {stdout.decode()}")
            logger.error(f"Subprocess errors: {stderr.decode()}")
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("Subprocess timed out and was killed.")

        if monitor.current_pose:
            final_positions.append((monitor.current_pose.x, monitor.current_pose.y, monitor.current_pose.z))
            final_euler_angles.append(monitor.final_angles)
            logger.info(f"Final position recorded: {monitor.current_pose}")
        else:
            final_positions.append((None, None, None))
            final_euler_angles.append((None, None, None))
            logger.warning("No final position recorded; appending None values.")

    return "Process completed successfully" if process.returncode == 0 else "Process terminated"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulations with adjustable timeout.")
    parser.add_argument('--timeout', type=int, default=150, help="Timeout duration in seconds.")
    args = parser.parse_args()

    rospy.init_node('simulation_monitor', anonymous=True)
    monitor = SimulationMonitor()

    try:
        max_xy_offset_cap = 50  # Maximum allowable offset for x and y
        rate = rospy.Rate(10)  # 10 Hz rate for the main loop

        for i in range(10):
            # Uniformly sample z
            init_z = np.random.uniform(z_min, z_max)

            # Calculate z offset
            z_offset = abs(init_z - ideal_z)

            # Dynamically calculate x and y offsets, capped at max_xy_offset_cap
            max_xy_offset = min(z_offset, max_xy_offset_cap)

            # Calculate valid ranges for x and y
            x_range = (ideal_x - max_xy_offset, ideal_x + max_xy_offset)
            y_range = (ideal_y - max_xy_offset, ideal_y + max_xy_offset)

            # Ensure x and y remain within their respective bounds
            init_x = np.random.uniform(max(x_min, x_range[0]), min(x_max, x_range[1]))
            init_y = np.random.uniform(max(y_min, y_range[0]), min(y_max, y_range[1]))

            # Append to initial positions
            initial_positions.append((init_x, init_y, init_z))

            logger.info(f"Starting iteration {i + 1}...")
            start_time = time.time()
            stop_reason = run_simulation(monitor, init_x, init_y, init_z, args.timeout)
            landing_times.append(time.time() - start_time)
            logger.info(f"Iteration {i + 1} stopped. Reason: {stop_reason}")

            success = (
                monitor.current_pose and
                abs(monitor.current_pose.z - ideal_z) < 1 and
                abs(monitor.current_pose.x - ideal_x) <= 4 and
                abs(monitor.current_pose.y - ideal_y) <= 4
            )
            landing_results.append("Success" if success else "Fail")

            monitor.reset()
            rate.sleep()

    except KeyboardInterrupt:
        rospy.signal_shutdown("KeyboardInterrupt")
        os._exit(0)

    # Save results after simulation loop
    run_folder = create_run_folder()
    save_results_to_csv(run_folder, initial_positions, final_positions, landing_times, landing_results, final_euler_angles)
    print(f"Plots saved in {run_folder}")

    save_summary_to_csv_and_metadata(base_dir="runs")
    print(f"Performance summary saved.")

    # Update simulation status to 'True' after results are saved
    with open(constants.simulation_status_file, 'w') as status_file:
        status_file.write('True')

    logger.info("Simulation status updated to True.")
