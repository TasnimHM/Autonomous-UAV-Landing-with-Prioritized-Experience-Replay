#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Vector3
from std_msgs.msg import Header

class LandingPathNode:
    def __init__(self):
        rospy.init_node("node_landing_path")

        # --- Parameters ---
        self.center_tolerance = rospy.get_param("~center_tolerance", 20)  # pixels
        self.stable_time = rospy.get_param("~stable_time", 3.0)  # seconds of stability before descent
        self.descent_levels = [100.0, 50.0, 5.0]  # Z-levels for descent
        self.publish_rate = rospy.get_param("~publish_rate", 10)

        # --- Subscribers ---
        self.sub_tgt_box = rospy.Subscriber(
            "/controller_node/tgt_box_rcvd", Vector3, self.cb_tgt_box, queue_size=1
        )
        self.sub_pose = rospy.Subscriber(
            "/jaxguam/pose", PoseStamped, self.cb_pose, queue_size=1
        )

        # --- Publisher ---
        self.pub_reference = rospy.Publisher("/planner/reference", PoseStamped, queue_size=3)

        # --- State ---
        self.current_pose = None
        self.last_centered_time = None
        self.target_locked = False
        self.x_center_img = 224
        self.y_center_img = 224
        self.center_error_x = 0
        self.center_error_y = 0

        rospy.loginfo("[node_landing_path] Initialized and waiting for centering...")
        self.run()

    # --- Callbacks ---
    def cb_tgt_box(self, msg: Vector3):
        """Receives the target box center from controller"""
        self.center_error_x = msg.x - self.x_center_img
        self.center_error_y = msg.y - self.y_center_img

        # Check if within tolerance
        if abs(self.center_error_x) < self.center_tolerance and abs(self.center_error_y) < self.center_tolerance:
            if self.last_centered_time is None:
                self.last_centered_time = rospy.Time.now()
        else:
            # Reset timer if out of center
            self.last_centered_time = None

    def cb_pose(self, msg: PoseStamped):
        self.current_pose = msg

    # --- Main Loop ---
    def run(self):
        rate = rospy.Rate(self.publish_rate)
        while not rospy.is_shutdown():
            if self.current_pose is None:
                rate.sleep()
                continue

            # Check if centered long enough
            if (
                self.last_centered_time is not None
                and not self.target_locked
                and (rospy.Time.now() - self.last_centered_time).to_sec() > self.stable_time
            ):
                self.target_locked = True
                rospy.loginfo("[node_landing_path] UAV centered — generating descent path...")
                self.publish_descent_path()

            rate.sleep()

    # --- Generate descent path ---
    def publish_descent_path(self):
        """Publishes waypoints for smooth vertical descent once target is centered."""
        x_curr = self.current_pose.pose.position.x
        y_curr = self.current_pose.pose.position.y
        z_curr = self.current_pose.pose.position.z

        for z_target in self.descent_levels:
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position.x = x_curr
            pose_msg.pose.position.y = y_curr
            pose_msg.pose.position.z = z_target

            # No rotation needed
            pose_msg.pose.orientation.x = 0.0
            pose_msg.pose.orientation.y = 0.0
            pose_msg.pose.orientation.z = 0.0
            pose_msg.pose.orientation.w = 1.0

            self.pub_reference.publish(pose_msg)
            rospy.loginfo(f"[node_landing_path] Published waypoint → (x={x_curr:.2f}, y={y_curr:.2f}, z={z_target:.2f})")
            rospy.sleep(2.0)  # small delay between waypoints

        rospy.loginfo("[node_landing_path] Descent path published successfully.")


if __name__ == "__main__":
    try:
        LandingPathNode()
    except rospy.ROSInterruptException:
        pass
