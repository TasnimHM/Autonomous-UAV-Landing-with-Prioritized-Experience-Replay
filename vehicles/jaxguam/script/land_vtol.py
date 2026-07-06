#!/usr/bin/env python3

import functools as ft
import os
import numpy as np
import rospy
import ipdb
import argparse
from loguru import logger
from geometry_msgs.msg import PoseStamped, Vector3, Twist

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.guam_types import RefInputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format

from guam_plot_batch_with_ref import plot_batch_with_ref
from utils.vehicle import Vehicle_Node
from utils.config import load_yaml_file
from utils import constants


class GUAM_Node(Vehicle_Node):
    def __init__(self, config):
        self.config = config
        super(GUAM_Node, self).__init__(config)

        jax_use_cpu()
        jax_use_double()
        set_logger_format()

        # Simulation parameters
        self.skip_sleep = config['ego_vehicle']['skip_sleep']
        self.plot_switch = config['ego_vehicle']['plot']
        self.save_video = config['ego_vehicle']['save_video']

        # GUAM-related parameters
        self.guam = None
        self.guam_reference = None
        self.control_msg = None
        self.kk = 0

        logger.info("Subscribing to planner for trajectory reference...")
        
        self.guam_disp_velCMD_sub = rospy.Subscriber('/display_node/control_cmd', Twist, self.guam_velocity_cmd_callback)        
        
        self.guam_control_velCMD_sub = rospy.Subscriber(config['ego_vehicle']['planner_topic'], Vector3, self.guam_control_cmd_callback)

        self.guam_reference_init()
        while self.guam_reference is None:
            pass

    def guam_control_cmd_callback(self, msg):
        self.control_msg = msg

    def guam_reference_init(self):
        # Adjust for GUAM units and frame
        pos_des = jnp.array(self.initial_position) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array(self.initial_velocity) * jnp.array([3.28084, -3.28084, -3.28084])
        chi_des = 0
        chi_dot_des = 0
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def guam_velocity_cmd_callback(self, msg):
        # Initial condition
        if self.kk == 0:
            vel_bIc_des = self.initial_velocity
            chi_des=0
            chi_dot_des = 0
            pos_des = self.initial_position
        # After initialization
        else:
            vel_bIc_des = np.array([msg.linear.x, msg.linear.y, msg.linear.z]) * 10
            
            if self.control_msg is not None:
                vel_bIc_des += np.array([self.control_msg.x * -1, self.control_msg.y, self.control_msg.z])

            pos_des = np.array([
                self.vehicle_pose_msg.pose.position.x,
                self.vehicle_pose_msg.pose.position.y,
                self.vehicle_pose_msg.pose.position.z
            ]) 
            pos_des += vel_bIc_des * 0.005  # in Hz, GUAM's dt = 0.005
        
            chi_des= 0
            chi_dot_des = 0

        # Convert meters to feet
        pos_des = jnp.array(pos_des) * jnp.array([3.28084, 3.28084, -3.28084])
        vel_bIc_des = jnp.array(vel_bIc_des) * jnp.array([3.28084, -3.28084, -3.28084])
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=chi_des,
            Chi_dot_des=chi_dot_des,
        )

    def main(self):
        logger.info("Constructing GUAM...")
        self.guam = FuncGUAM()
        batch_size = 1  # Simulate for a single vehicle test
        state = GuamState.create()

        # Initialize GUAM state
        state.aircraft[:] = np.array([
            self.guam_reference.Vel_bIc_des[0],
            self.guam_reference.Vel_bIc_des[1],
            self.guam_reference.Vel_bIc_des[2],
            0, 0, 0,
            self.guam_reference.Pos_des[0],
            self.guam_reference.Pos_des[1],
            self.guam_reference.Pos_des[2],
            1, 0, 0, 0
        ])

        logger.info("Calling GUAM...")
        b_state = jtu.tree_map(lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state)

        vmap_step = jax.jit(jax.vmap(ft.partial(self.guam.step, self.guam.dt), in_axes=(0, None)))
        loop_rate = rospy.Rate(1 / self.guam.dt)  # in Hz, GUAM's dt = 0.005

        def simulate_batch(self, b_state0) -> GuamState:
            b_state = b_state0
            if self.plot_switch:
                Tb_state = [b_state0]
                time_list = [0]
                Ref_list = [b_state0.aircraft[0][0:3].tolist() + b_state0.aircraft[0][6:9].tolist()]

            self.kk = 0
            while not rospy.is_shutdown():
                t = self.kk * self.guam.dt
                self.kk += 1

                ref_inputs = self.guam_reference
                b_state = vmap_step(b_state, ref_inputs)

                if self.plot_switch:
                    time_list.append(t)
                    Ref_list.append(ref_inputs.Vel_bIc_des.tolist() + ref_inputs.Pos_des.tolist())
                    Tb_state.append(jax2np(b_state))

                self.publish_state(b_state)

                if not self.skip_sleep:
                    loop_rate.sleep()

            if self.plot_switch:
                bT_state = jtu.tree_map(lambda *args: np.stack(list(args), axis=1), *Tb_state)
                return bT_state, Ref_list, time_list

        if self.plot_switch:
            if not os.path.exists("results"):
                os.makedirs("results")
            bT_state, Ref_list, time_list = simulate_batch(self, b_state)
            np.savez("results/bT_state.npz", aircraft=bT_state.aircraft)
            Ref_list = np.array(Ref_list)
            np.savez("results/Ref_list.npz", Vel_des=Ref_list[:, 0:3], Pos_des=Ref_list[:, 3:6])
            np.savez("results/time_list.npz", np.array(time_list))
        else:
            simulate_batch(self, b_state)

    def publish_state(self, b_state):
        # Convert position back to meters for publishing
        self.vehicle_pose_msg.pose.position.x = b_state.aircraft[0][6] / 3.28084
        self.vehicle_pose_msg.pose.position.y = b_state.aircraft[0][7] / 3.28084
        self.vehicle_pose_msg.pose.position.z = b_state.aircraft[0][8] / 3.28084 * -1
        self.vehicle_pose_msg.pose.orientation.x = b_state.aircraft[0][9]
        self.vehicle_pose_msg.pose.orientation.y = b_state.aircraft[0][10]
        self.vehicle_pose_msg.pose.orientation.z = b_state.aircraft[0][11]
        self.vehicle_pose_msg.pose.orientation.w = b_state.aircraft[0][12]
        self.vehicle_pose_pub.publish(self.vehicle_pose_msg)

        self.vehicle_vel_msg.linear.x = b_state.aircraft[0][0] / 3.28084
        self.vehicle_vel_msg.linear.y = b_state.aircraft[0][1] / 3.28084
        self.vehicle_vel_msg.linear.z = b_state.aircraft[0][2] / 3.28084 * -1
        self.vehicle_vel_pub.publish(self.vehicle_vel_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GUAM Node with dynamic config")
    parser.add_argument('--config', type=str, default=constants.merged_config_path, 
                        help='Path to the configuration file')
    args = parser.parse_args()

    config = load_yaml_file(args.config, __file__)
    vehicle_type = config['ego_vehicle']['type']
    assert vehicle_type == 'jaxguam', "This node only supports JaxGUAM vehicle."

    if config['ego_vehicle']['debug']:
        with ipdb.launch_ipdb_on_exception():
            guam_node = GUAM_Node(config)
            guam_node.main()
    else:
        guam_node = GUAM_Node(config)
        guam_node.main()

    if vehicle_type == 'jaxguam' and config['ego_vehicle']['plot']:
        logger.info("Plotting...")
        plot_batch_with_ref(save_video=config['ego_vehicle']['save_video'])
