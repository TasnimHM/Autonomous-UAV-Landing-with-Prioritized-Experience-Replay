#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped
from utils.config import load_yaml_file
from utils import constants


class SimplePlanner:
    """
    A lightweight planner that forwards controller Twist commands
    directly to GUAM's pose target topics.
    """
    def __init__(self, config):
        self.config = config
        rospy.init_node("planner")

        # --- Subscribers ---
        self.vel_sub = rospy.Subscriber(
            config['ego_vehicle']['perception_vel_topic'],
            Twist,
            self.velocity_callback
        )

        # --- Publishers ---
        self.target_pose_pub = rospy.Publisher('/target/pose', Twist, queue_size=1)
        self.target_waypoint_pub = rospy.Publisher('/target/waypoint', Float32MultiArray, queue_size=1)

        rospy.loginfo("🧭 SimplePlanner running — forwarding controller velocity to GUAM")

    def velocity_callback(self, msg: Twist):
        """Forward velocity commands directly to GUAM."""
        vx, vy, vz = msg.linear.x, msg.linear.y, msg.linear.z

        # Publish to pose
        pose_msg = Twist()
        pose_msg.linear.x = vx
        pose_msg.linear.y = vy
        pose_msg.linear.z = vz
        self.target_pose_pub.publish(pose_msg)

        # Publish to waypoint (optional debug/log)
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = [0, 0, 0, vx, vy, vz]
        self.target_waypoint_pub.publish(waypoint_msg)

        rospy.loginfo(f"[planner] Forwarded vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    config = load_yaml_file(constants.merged_config_path, __file__)
    planner = SimplePlanner(config)
    planner.run()
