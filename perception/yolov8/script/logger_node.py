#!/usr/bin/env python3
import rospy
import csv
import os
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32MultiArray

class LandingLogger:
    def __init__(self):
        rospy.init_node("landing_logger", anonymous=True)

        # ---- CONFIG ----
        self.det_topic = rospy.get_param("~det_topic", "/yolo_node/sort_large_predictions")

        save_dir = "/catkin_ws/src/yolov5/script/logs/large1_log"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"log_{rospy.Time.now().to_nsec()}.csv"
        self.filepath = os.path.join(save_dir, filename)

        self.csvfile = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.csvfile)

        # -------- HEADER --------
        self.writer.writerow([
            "time",
            "pos_x", "pos_y", "pos_z",
            "bb_x1", "bb_y1", "bb_x2", "bb_y2",
            "center_x", "center_y", "bbox_size",
            "track_id",
            "vx", "vy", "vz",
            "start_x", "start_y", "start_z"
        ])

        # Storage
        self.pose = None
        self.last_vx = 0.0
        self.last_vy = 0.0
        self.last_vz = 0.0

        # NEW: Starting position storage
        self.start_x = None
        self.start_y = None
        self.start_z = None

        # ---- SUBSCRIBERS ----
        rospy.Subscriber("/jaxguam/pose", PoseStamped, self.cb_pose)
        rospy.Subscriber(self.det_topic, Float32MultiArray, self.cb_det)
        rospy.Subscriber("/controller_node/vel_cmd", Twist, self.cb_vel)

        rospy.loginfo(f"[logger] Logging to: {self.filepath}")
        rospy.loginfo(f"[logger] Detection topic: {self.det_topic}")

    # ------------------- CALLBACKS -------------------

    def cb_pose(self, msg):
        self.pose = msg

        # Capture start position ONCE
        if self.start_x is None:
            self.start_x = msg.pose.position.x
            self.start_y = msg.pose.position.y
            self.start_z = msg.pose.position.z
            rospy.loginfo(f"[logger] Start pos = ({self.start_x:.2f}, {self.start_y:.2f}, {self.start_z:.2f})")

    def cb_vel(self, msg):
        self.last_vx = msg.linear.x
        self.last_vy = msg.linear.y
        self.last_vz = msg.linear.z

    def cb_det(self, msg):
        if self.pose is None:
            return

        data = msg.data
        if len(data) < 5:
            return

        # Format: [x1, y1, x2, y2, track_id]
        x1, y1, x2, y2, track_id = data[:5]

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bbox_size = (x2 - x1) * (y2 - y1)

        # Write row (same data, plus start positions)
        self.writer.writerow([
            rospy.get_time(),
            self.pose.pose.position.x,
            self.pose.pose.position.y,
            self.pose.pose.position.z,
            x1, y1, x2, y2,
            center_x, center_y, bbox_size,
            track_id,
            self.last_vx, self.last_vy, self.last_vz,
            self.start_x, self.start_y, self.start_z,
        ])

    # -------------------

    def spin(self):
        try:
            rospy.spin()
        finally:
            self.csvfile.close()
            rospy.loginfo(f"[logger] Saved log to: {self.filepath}")


if __name__ == "__main__":
    logger = LandingLogger()
    logger.spin()
