#!/usr/bin/env python3

import math
import time
import rospy
import heapq
import numpy as np
from typing import List
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, PoseStamped

from utils.config import load_yaml_file
from utils import constants


def compute_velocities(points, velocity_magnitude=5.0):
    velocities = []
    if len(points) <= 1:
        return [np.zeros(3)]
    
    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            velocities.append(np.zeros(3))
        else:
            direction = points[i + 1] - points[i]
            direction_norm = np.linalg.norm(direction)
            if direction_norm != 0:
                velocity_vector = (direction / direction_norm) * velocity_magnitude
            else:
                velocity_vector = np.zeros(3)
            velocities.append(velocity_vector)
    return velocities


class PathPlanner:
    """
    Path planner class. It utilizes the starting point and the target point to calculate the path.
    """
    def __init__(self, config) -> None:
        self.config = config

        rospy.init_node("planner")  

        self.global_target_sub = rospy.Subscriber(config['ego_vehicle']['reference_topic'], Twist, self.global_target_callback)
        self.pose_sub = rospy.Subscriber(f"/{config['ego_vehicle']['type']}/pose", PoseStamped, self.pose_callback)

        if self.config['ego_vehicle']['planner'] == 'simple':
            self.perception_vel_sub = rospy.Subscriber(
                config['ego_vehicle']['perception_vel_topic'], 
                Twist, 
                self.perception_vel_callback
            )
            self.perception_control_sub = rospy.Subscriber(
                config['ego_vehicle']['perception_control_topic'], 
                Float32MultiArray, 
                self.perception_control_callback
            )
            rospy.loginfo("Subscribed to perception_vel_topic and perception_control_topic")

        # Publishers
        self.target_waypoint_pub = rospy.Publisher('/target/waypoint', Float32MultiArray, queue_size=1)
        self.target_pose_pub = rospy.Publisher('/target/pose', Twist, queue_size=1)  # <-- NEW

        self.start_point, self.end_point = self.get_start_end_points(config)
        self.waypoints, self.velocities = self.get_waypoints(self.start_point, self.end_point)
        self.waypoint_counter = 0

    def get_waypoints(self, start_point: np.array, end_point: np.array) -> List[np.array]:
        if self.config['ego_vehicle']['planner'] == 'simple':
            waypoints = [start_point]
            velocities = [np.ones(3)]
        elif self.config['ego_vehicle']['planner'] == 'a_star':
            waypoints = astar(tuple(start_point.tolist()), tuple(end_point.tolist()))
            velocities = compute_velocities([np.array(x) for x in waypoints], velocity_magnitude=0.5)
        else:
            raise ValueError(f"Unknown planner {self.config['ego_vehicle']['planner']}")
        return waypoints, velocities

    def get_start_end_points(self, config):
        start_point = np.array([
            config['ego_vehicle']['location']['x'],
            config['ego_vehicle']['location']['y'],
            config['ego_vehicle']['location']['z']
        ])
        end_point = np.array([
            config['target']['x'],
            config['target']['y'],
            config['target']['z']
        ])
        if config['target']['type'] == 'relative':
            end_point += start_point
        elif config['target']['type'] == 'absolute':
            pass
        else:
            raise ValueError(f"Unknown target type {config['target']['type']}.")
        return (start_point, end_point)

    def run(self):
        r = rospy.Rate(10)
        start_time = time.time()

        while not rospy.is_shutdown():
            message = Float32MultiArray()
            message.data = [
                self.waypoints[self.waypoint_counter][0],
                self.waypoints[self.waypoint_counter][1],
                self.waypoints[self.waypoint_counter][2],
                self.velocities[self.waypoint_counter][0],
                self.velocities[self.waypoint_counter][1],
                self.velocities[self.waypoint_counter][2]
            ]
            if time.time() - start_time < 5:
                message.data = [
                    self.config['ego_vehicle']['location']['x'],
                    self.config['ego_vehicle']['location']['y'],
                    self.config['ego_vehicle']['location']['z'],
                    0, 0, 0
                ]
            self.target_waypoint_pub.publish(message)
            r.sleep()
    
    def pose_callback(self, data):
        if self.waypoint_counter < len(self.waypoints) - 1:
            x_curr = data.pose.position.x
            y_curr = data.pose.position.y
            z_curr = data.pose.position.z
            x_waypoint = self.waypoints[self.waypoint_counter][0]
            y_waypoint = self.waypoints[self.waypoint_counter][1]
            z_waypoint = self.waypoints[self.waypoint_counter][2]
            dist = math.sqrt((x_curr - x_waypoint) ** 2 + (y_curr - y_waypoint) ** 2 + (z_curr - z_waypoint) ** 2)
            if dist < self.config['landing_threshold']:
                self.waypoint_counter += 1

    def global_target_callback(self, data):
        pass

    def perception_vel_callback(self, msg: Twist):
        """
        Forward perception velocity commands directly as waypoint and pose updates.
        """
        # Avoid all-stop (zero velocity) -> GUAM freeze
        vx, vy, vz = msg.linear.x, msg.linear.y, msg.linear.z
        eps = 1e-3
        if abs(vx) < eps and abs(vy) < eps and abs(vz) < eps:
            vz = -eps   # tiny downward motion to keep GUAM alive

        # Publish waypoint (debug/logging)
        message = Float32MultiArray()
        message.data = [0, 0, 0, msg.linear.x, msg.linear.y, msg.linear.z]
        self.target_waypoint_pub.publish(message)

        # Publish pose (for JAX GUAM)
        pose_msg = Twist()
        pose_msg.linear.x = msg.linear.x
        pose_msg.linear.y = msg.linear.y
        pose_msg.linear.z = msg.linear.z
        self.target_pose_pub.publish(pose_msg)

        rospy.loginfo(f"Forwarded perception velocity: vx={msg.linear.x:.2f}, vy={msg.linear.y:.2f}, vz={msg.linear.z:.2f}")

    def perception_control_callback(self, msg: Float32MultiArray):
        rospy.loginfo(f"Received perception control data (unused): {msg.data}")


if __name__ == "__main__":
    config = load_yaml_file(constants.merged_config_path, __file__)
    planner = PathPlanner(config=config)
    planner.run()
