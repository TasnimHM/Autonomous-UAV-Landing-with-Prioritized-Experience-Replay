#!/bin/bash
set -e

# setup ROS env
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

# start dual YOLO node (Large -> panel2, Small -> panel3, Small + SORT for controller)
rosrun yolov5 node_yolo_dual.py __name:=yolo_dual_node \
  _camera_topic:=/carla_node/cam_down/image_raw \
  _topic_large_img:=/yolo_node/yolo_large_frame \
  _topic_small_img:=/yolo_node/yolo_small_frame \
  _publish_arrays:=true \
  _topic_large_array:=/yolo_node/yolo_large_array \
  _topic_small_array:=/yolo_node/yolo_small_array

# keep container alive
tail -f /dev/null
