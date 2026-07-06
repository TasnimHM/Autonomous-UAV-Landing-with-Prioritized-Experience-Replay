#!/usr/bin/env python3
"""
episode_manager.py — Randomized episode reset manager for PER data collection.

Flow per episode:
  1. Sample a random (x, y, z) initial pose from the distributions used in the paper.
  2. Publish it on /episode/initial_pose  (geometry_msgs/Pose).
  3. Set reset_called ROS param → triggers environment.reset() in node_carla.py.
  4. Publish /episode/reset → resets controller state in landing_controller.py.
  5. Wait for /episode/done.
  6. Log the result and move to next episode.

Pose ranges (matching the paper's Table IV / Fig. 4a):
  x0 ~ U(-95, -65)   [m]
  y0 ~ U( 60,  90)   [m]
  z0 ∈ {70, 80, 90, 110}  [m]   (discrete altitudes)
"""

import random
import numpy as np
import rospy

from std_msgs.msg import Bool, Empty, Int32
from geometry_msgs.msg import Pose


# ── Randomization ranges (metres, CARLA world frame) ─────────────────────────
X_MIN, X_MAX   = -90.0, -70.0
Y_MIN, Y_MAX   =  65.0,  85.0
Z_OPTIONS      = [70, 80, 90]   # discrete altitude pool


def sample_initial_pose() -> Pose:
    """Sample a random starting pose matching the paper's experimental setup."""
    pose = Pose()
    pose.position.x = random.uniform(X_MIN, X_MAX)
    pose.position.y = random.uniform(Y_MIN, Y_MAX)
    pose.position.z = float(random.choice(Z_OPTIONS))
    # Zero rotation — UAV starts level
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = 0.0
    pose.orientation.w = 1.0
    return pose


class EpisodeManager:

    def __init__(self):
        rospy.init_node("episode_manager")

        self.total_episodes = rospy.get_param("~episodes", 21)
        self.current_episode = 0
        self.done_flag = False

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_reset        = rospy.Publisher("/episode/reset",        Empty, queue_size=1)
        self.pub_id           = rospy.Publisher("/episode/id",           Int32, queue_size=1)
        self.pub_initial_pose = rospy.Publisher("/episode/initial_pose", Pose,  queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber("/episode/done", Bool, self._done_callback)

        # Give ROS time to connect publishers before the first episode
        rospy.sleep(2.0)
        rospy.loginfo("[episode_manager] Ready — will run %d episodes.", self.total_episodes)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _done_callback(self, msg: Bool):
        if msg.data:
            self.done_flag = True

    # ── Reset helpers ─────────────────────────────────────────────────────────

    def _publish_initial_pose(self, pose: Pose):
        """
        Publish the desired initial pose several times so that both
        node_vehicle.py (GUAM) and node_carla.py (CARLA actor) receive it
        even if they missed the first message.
        """
        for _ in range(5):
            self.pub_initial_pose.publish(pose)
            rospy.sleep(0.05)

    def _request_environment_reset(self, pose: Pose):
        """
        1. Tell GUAM/CARLA the target start pose.
        2. Flip the reset_called param that node_carla.py polls.
        """
        self._publish_initial_pose(pose)
        rospy.set_param("reset_called", True)
        rospy.loginfo(
            "[episode_manager] Reset requested → x=%.1f  y=%.1f  z=%.1f",
            pose.position.x, pose.position.y, pose.position.z,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(2)

        while self.current_episode < self.total_episodes and not rospy.is_shutdown():

            rospy.loginfo("🚀 Starting episode %d / %d",
                          self.current_episode, self.total_episodes)

            # ── Sample a new random start pose ────────────────────────────────
            pose = sample_initial_pose()
            #pose = Pose()
            #pose.position.x = -80.0
            #pose.position.y = 75.0
            #pose.position.z = 70.0
            #pose.orientation.w = 1.0
            rospy.loginfo(
                "   Sampled pose → x=%.2f  y=%.2f  z=%.2f",
                pose.position.x, pose.position.y, pose.position.z,
            )

            self.done_flag = False

            # 1. Broadcast episode id
            self.pub_id.publish(Int32(data=self.current_episode))
            rospy.sleep(0.2)

            # 2. Broadcast pose + trigger CARLA/GUAM reset
            self._request_environment_reset(pose)
            rospy.sleep(3.0)   # allow environment.reset() to complete

            # 3. Reset controller internal state
            self.pub_reset.publish(Empty())
            rospy.sleep(0.5)

            # 4. Wait for landing to finish (or timeout)
            timeout = rospy.Duration(120.0)   # 2-minute safety timeout
            t_start = rospy.Time.now()
            while not self.done_flag and not rospy.is_shutdown():
                if (rospy.Time.now() - t_start) > timeout:
                    rospy.logwarn("[episode_manager] ⏱️  Episode %d timed out — forcing next.",
                                  self.current_episode)
                    break
                rate.sleep()

            rospy.loginfo("✅ Episode %d complete.", self.current_episode)
            self.current_episode += 1

        rospy.loginfo("🎉 All %d episodes finished.", self.total_episodes)


if __name__ == "__main__":
    manager = EpisodeManager()
    manager.run()
    