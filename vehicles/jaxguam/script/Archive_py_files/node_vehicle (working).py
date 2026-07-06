#!/usr/bin/env python3
"""
node_vehicle.py — GUAM vehicle node with hot-reset support.

Changes from original:
  • Subscribes to /episode/initial_pose (geometry_msgs/Pose).
  • When a new pose arrives, sets self._pending_reset so the simulate_batch
    loop can safely reinitialise b_state between GUAM steps (no thread-safety
    issues with JAX).
  • environment.reset() in node_carla.py handles the CARLA actor teleport;
    this node only needs to restart the GUAM integrator from the new position.
"""

import functools as ft
import numpy as np
import ipdb
import os
import rospy

from loguru import logger
from geometry_msgs.msg import PoseStamped, Vector3, Twist, Pose

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax_guam.functional.guam_new import FuncGUAM, GuamState
from jax_guam.guam_types import RefInputs
from jax_guam.utils.jax_utils import jax2np, jax_use_cpu, jax_use_double
from jax_guam.utils.logging import set_logger_format

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
        self.skip_sleep  = config['ego_vehicle']['skip_sleep']
        self.plot_switch = config['ego_vehicle']['plot']
        self.save_video  = config['ego_vehicle']['save_video']

        # GUAM state
        self.guam           = None
        self.guam_reference = None
        self.control_msg    = None
        self.kk             = 0

        # ── Reset support ─────────────────────────────────────────────────────
        # Written by the subscriber callback; consumed (and cleared) inside the
        # simulate_batch loop so we never touch JAX arrays from two threads.
        self._pending_reset      = False
        self._pending_reset_pose = None   # geometry_msgs/Pose or None

        logger.info("Subscribing to planner for trajectory reference...")

        # Subscribe to controller velocity commands
        self.guam_disp_velCMD_sub = rospy.Subscriber(
            '/controller_node/vel_cmd',
            Twist,
            self.guam_velocity_cmd_callback,
        )

        # Subscribe to planner position control
        self.guam_control_velCMD_sub = rospy.Subscriber(
            config['ego_vehicle']['planner_topic'],
            Vector3,
            self.guam_control_cmd_callback,
        )

        # ── NEW: listen for randomized initial pose from episode_manager ──────
        rospy.Subscriber(
            '/episode/initial_pose',
            Pose,
            self._initial_pose_callback,
        )

        self.guam_reference_init()
        while self.guam_reference is None:
            pass

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def guam_control_cmd_callback(self, msg):
        self.control_msg = msg

    def _initial_pose_callback(self, msg: Pose):
        """
        Called when episode_manager publishes a new starting pose.
        We just store it — the simulate_batch loop applies it safely.
        """
        self._pending_reset_pose = msg
        self._pending_reset      = True
        logger.info(
            "[GUAM] Reset queued → x={:.2f}  y={:.2f}  z={:.2f}",
            msg.position.x, msg.position.y, msg.position.z,
        )

    # ── GUAM reference helpers ────────────────────────────────────────────────

    def guam_reference_init(self):
        """Initialise from the static YAML config (first episode)."""
        pos_des      = jnp.array(self.initial_position) * jnp.array([3.28084,  3.28084, -3.28084])
        vel_bIc_des  = jnp.array(self.initial_velocity) * jnp.array([3.28084, -3.28084, -3.28084])
        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des,
            Pos_des=pos_des,
            Chi_des=0,
            Chi_dot_des=0,
        )

    def _make_reference_from_pose(self, pose: Pose) -> RefInputs:
        """Convert a geometry_msgs/Pose into a GUAM RefInputs (feet, NED)."""
        pos_m = np.array([pose.position.x, pose.position.y, pose.position.z])
        vel_m = np.zeros(3)   # start from rest

        pos_ft     = jnp.array(pos_m) * jnp.array([3.28084,  3.28084, -3.28084])
        vel_bIc_ft = jnp.array(vel_m) * jnp.array([3.28084, -3.28084, -3.28084])

        return RefInputs(
            Vel_bIc_des=vel_bIc_ft,
            Pos_des=pos_ft,
            Chi_des=0,
            Chi_dot_des=0,
        )

    def _make_b_state_from_reference(self, ref: RefInputs, batch_size: int) -> "GuamState":
        """Create a fresh GUAM batch state initialised to ref position/velocity."""
        state = GuamState.create()
        state.aircraft[:] = np.array([
            ref.Vel_bIc_des[0], ref.Vel_bIc_des[1], ref.Vel_bIc_des[2],
            0, 0, 0,
            ref.Pos_des[0], ref.Pos_des[1], ref.Pos_des[2],
            1, 0, 0, 0
        ])
        b_state = jtu.tree_map(
            lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state
        )
        return b_state

    def guam_velocity_cmd_callback(self, msg: Twist):
        rospy.loginfo(
            f"[GUAM-DEBUG] raw msg: x={msg.linear.x:.3f}, "
            f"y={msg.linear.y:.3f}, z={msg.linear.z:.3f}"
        )

        vel_bIc_des = np.array([msg.linear.x, msg.linear.y, msg.linear.z])

        if self.guam_reference is not None:
            pos_des = np.array(self.guam_reference.Pos_des) / np.array([3.28084, 3.28084, -3.28084])
        else:
            pos_des = np.array(self.initial_position)

        dt = getattr(self.guam, "dt", 0.005)
        pos_des += vel_bIc_des * dt

        vel_bIc_des_ft = jnp.array(vel_bIc_des) * jnp.array([3.28084, -3.28084, -3.28084])
        pos_des_ft     = jnp.array(pos_des)      * jnp.array([3.28084,  3.28084, -3.28084])

        self.guam_reference = RefInputs(
            Vel_bIc_des=vel_bIc_des_ft,
            Pos_des=pos_des_ft,
            Chi_des=0,
            Chi_dot_des=0,
        )

        rospy.loginfo(
            f"[GUAM] vx={vel_bIc_des[0]:.2f}, vy={vel_bIc_des[1]:.2f}, "
            f"vz={vel_bIc_des[2]:.2f}, "
            f"pos=({pos_des[0]:.2f}, {pos_des[1]:.2f}, {pos_des[2]:.2f})"
        )

    # ── Main simulation loop ──────────────────────────────────────────────────

    def main(self):
        logger.info("Constructing GUAM...")
        self.guam = FuncGUAM()
        batch_size = 1
        state = GuamState.create()

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
        b_state = jtu.tree_map(
            lambda x: np.broadcast_to(x, (batch_size,) + x.shape).copy(), state
        )

        vmap_step  = jax.jit(jax.vmap(ft.partial(self.guam.step, self.guam.dt), in_axes=(0, None)))
        loop_rate  = rospy.Rate(1 / self.guam.dt)

        def simulate_batch(self, b_state0):
            b_state = b_state0

            if self.plot_switch:
                Tb_state  = [b_state0]
                time_list = [0]
                Ref_list  = [
                    b_state0.aircraft[0][0:3].tolist() +
                    b_state0.aircraft[0][6:9].tolist()
                ]

            self.kk = 0

            while not rospy.is_shutdown():
                t = self.kk * self.guam.dt
                self.kk += 1

                # ── HOT RESET ────────────────────────────────────────────────
                # Check once per step; cost is negligible (just a bool read).
                if self._pending_reset and self._pending_reset_pose is not None:
                    new_pose  = self._pending_reset_pose
                    new_ref   = self._make_reference_from_pose(new_pose)
                    b_state   = self._make_b_state_from_reference(new_ref, batch_size)

                    # Update the running reference so velocity callbacks have
                    # a consistent base position.
                    self.guam_reference      = new_ref
                    self._pending_reset      = False
                    self._pending_reset_pose = None
                    self.kk                  = 0

                    logger.info(
                        "[GUAM] ✅ Hot-reset applied → pos_ft=({:.1f}, {:.1f}, {:.1f})",
                        new_ref.Pos_des[0], new_ref.Pos_des[1], new_ref.Pos_des[2],
                    )

                    if self.plot_switch:
                        Tb_state  = [b_state]
                        time_list = [0]
                        Ref_list  = [
                            b_state.aircraft[0][0:3].tolist() +
                            b_state.aircraft[0][6:9].tolist()
                        ]

                    # Publish the new state immediately so CARLA and the
                    # controller see the teleported position right away.
                    self.publish_state(b_state)
                    if not self.skip_sleep:
                        loop_rate.sleep()
                    continue
                # ─────────────────────────────────────────────────────────────

                ref_inputs = self.guam_reference
                b_state    = vmap_step(b_state, ref_inputs)

                if self.plot_switch:
                    time_list.append(t)
                    Ref_list.append(
                        ref_inputs.Vel_bIc_des.tolist() +
                        ref_inputs.Pos_des.tolist()
                    )
                    Tb_state.append(jax2np(b_state))

                self.publish_state(b_state)

                if not self.skip_sleep:
                    loop_rate.sleep()

            if self.plot_switch:
                bT_state = jtu.tree_map(
                    lambda *args: np.stack(list(args), axis=1), *Tb_state
                )
                return bT_state, Ref_list, time_list

        if self.plot_switch:
            if not os.path.exists("results"):
                os.makedirs("results")
            bT_state, Ref_list, time_list = simulate_batch(self, b_state)
            np.savez("results/bT_state.npz",  aircraft=bT_state.aircraft)
            Ref_list = np.array(Ref_list)
            np.savez("results/Ref_list.npz",
                     Vel_des=Ref_list[:, 0:3], Pos_des=Ref_list[:, 3:6])
            np.savez("results/time_list.npz", np.array(time_list))
        else:
            simulate_batch(self, b_state)

    def publish_state(self, b_state):
        # Position: feet → metres, NED → ENU
        self.vehicle_pose_msg.pose.position.x =  b_state.aircraft[0][6] / 3.28084
        self.vehicle_pose_msg.pose.position.y =  b_state.aircraft[0][7] / 3.28084
        self.vehicle_pose_msg.pose.position.z =  b_state.aircraft[0][8] / 3.28084 * -1

        # Quaternion: GUAM NED → CARLA ENU correction
        q_x = b_state.aircraft[0][9]
        q_y = b_state.aircraft[0][10]
        q_z = b_state.aircraft[0][11]
        q_w = b_state.aircraft[0][12]

        self.vehicle_pose_msg.pose.orientation.x = -q_w
        self.vehicle_pose_msg.pose.orientation.y = -q_z
        self.vehicle_pose_msg.pose.orientation.z =  q_y
        self.vehicle_pose_msg.pose.orientation.w =  q_x

        self.vehicle_pose_pub.publish(self.vehicle_pose_msg)

        # Velocity: ft/s → m/s, invert z
        self.vehicle_vel_msg.linear.x =  b_state.aircraft[0][0] / 3.28084
        self.vehicle_vel_msg.linear.y =  b_state.aircraft[0][1] / 3.28084
        self.vehicle_vel_msg.linear.z =  b_state.aircraft[0][2] / 3.28084 * -1
        self.vehicle_vel_pub.publish(self.vehicle_vel_msg)


if __name__ == "__main__":
    config       = load_yaml_file(constants.merged_config_path, __file__)
    vehicle_type = config['ego_vehicle']['type']
    assert vehicle_type == 'jaxguam', "This node only supports JaxGUAM vehicle."

    if config['ego_vehicle']['debug']:
        with ipdb.launch_ipdb_on_exception():
            guam_node = GUAM_Node(config)
            guam_node.main()
    else:
        guam_node = GUAM_Node(config)
        guam_node.main()