#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist, PoseStamped, Vector3

class MoveAlongZNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('move_along_z_node', anonymous=True)
        
        # Create publishers for velocity and pose
        self.velocity_pub = rospy.Publisher('/jaxguam/velocity', Twist, queue_size=10)
        self.pose_pub = rospy.Publisher('/jaxguam/pose', PoseStamped, queue_size=10)
        
        # Create a subscriber for velocity command
        rospy.Subscriber('/controller_node/vel_cmd', Vector3, self.vel_cmd_callback)
        
        # Initialize current and previous velocities
        self.current_velocity = Vector3(x=0.0, y=0.0, z=1.0)  # Initial z velocity is 1.0 m/s
        self.previous_velocity = Twist()
        
        # Define the rate of publishing (10 Hz)
        self.rate = rospy.Rate(20)
        
        # Initialize position
        self.pose_msg = PoseStamped()
        self.pose_msg.header.frame_id = "world"  # Frame of reference
        self.pose_msg.pose.position.x = -80.0
        self.pose_msg.pose.position.y = 70.0
        self.pose_msg.pose.position.z = 75.0
        self.pose_msg.pose.orientation.w = 1.0  # Default orientation
        
        # Set z-target for stopping
        self.z_target = 35.0

    def vel_cmd_callback(self, msg):
    #"""Callback to update velocity incrementally from /controller_node/vel_cmd."""
        if msg.x == 0.0 and msg.y == 0.0 and msg.z == 0.0:
            # If all components of the velocity message are zero, reset velocities to zero
            self.current_velocity.x = 0.0
            self.current_velocity.y = 0.0
            self.current_velocity.z = 0.0
            # rospy.loginfo("Received zero velocity message. All velocities set to zero.")
        else:
            # Otherwise, increment the velocities as usual
            self.current_velocity.x += msg.x
            self.current_velocity.y += msg.y
            self.current_velocity.z += msg.z  # Increment z velocity
            # rospy.loginfo(f"Received new velocity increment: x={msg.x}, y={msg.y}, z={msg.z}")

    def move(self):
        try:
            while not rospy.is_shutdown() and self.pose_msg.pose.position.z > self.z_target:
                # Update positions based on the latest velocity and time step
                self.pose_msg.pose.position.x += self.current_velocity.x * 0.001
                self.pose_msg.pose.position.y += self.current_velocity.y * 0.001
                self.pose_msg.pose.position.z += self.current_velocity.z * 0.001
                
                # Update the velocity message with the new calculated velocity
                move_msg = Twist()
                move_msg.linear.x = self.current_velocity.x
                move_msg.linear.y = self.current_velocity.y
                move_msg.linear.z = self.current_velocity.z
                
                # Publish updated pose and velocity
                self.velocity_pub.publish(move_msg)
                self.pose_pub.publish(self.pose_msg)
                
                # rospy.loginfo(f"Published velocity: {move_msg}")
                # rospy.loginfo(f"Published pose: {self.pose_msg}")
                
                self.rate.sleep()
        
        except rospy.ROSInterruptException:
            rospy.loginfo("Node interrupted with Ctrl+C")
        
        finally:
            # Stop the robot by publishing a zero Twist message
            stop_msg = Twist()  # Default zero-initialized message
            self.velocity_pub.publish(stop_msg)
            
            rospy.loginfo("Published stop command after interruption or reaching target z value.")
            rospy.loginfo(f"Final pose: {self.pose_msg}")

if __name__ == '__main__':
    try:
        node = MoveAlongZNode()
        node.move()
    except rospy.ROSInterruptException:
        rospy.loginfo("MoveAlongZNode terminated.")
