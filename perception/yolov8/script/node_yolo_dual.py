#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from ultralytics import YOLO

# --- SORT + helpers ---
from tools.msgpacking import init_matrix_array_ros_msg, pack_multiarray_ros_msg
from tools.sort import Sort


def detections_to_multiarray(ultra_result):
    """Convert Ultralytics result -> Float32MultiArray with rows [x1, y1, x2, y2, confidence]."""
    arr = Float32MultiArray()
    rows = []
    if ultra_result is not None and hasattr(ultra_result[0], "boxes"):
        for b in ultra_result[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            rows.extend([float(x1), float(y1), float(x2), float(y2), conf])
    if rows:
        n = len(rows) // 5
        arr.layout.dim = [
            MultiArrayDimension(label="rows", size=n, stride=n * 5),
            MultiArrayDimension(label="cols", size=5, stride=5),
        ]
    arr.data = rows
    return arr


def annotate_result(ultra_result, bgr_img, title_text=None):
    """Draw YOLO detections on frame."""
    vis = bgr_img.copy()
    if ultra_result is not None:
        try:
            vis = ultra_result[0].plot()
        except Exception:
            if hasattr(ultra_result[0], "boxes"):
                for b in ultra_result[0].boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    conf = float(b.conf[0])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis, f"helipad {conf:.2f}", (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if title_text:
        cv2.putText(vis, title_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return vis


class YoloDualNode(object):
    def __init__(self):
        # Params
        self.camera_topic = rospy.get_param("~camera_topic", "/carla_node/cam_down/image_raw")

        # weights
        self.small_weights = rospy.get_param("~small_weights", "/catkin_ws/src/yolov5/models/small.pt")
        self.large_weights = rospy.get_param("~large_weights", "/catkin_ws/src/yolov5/models/large.pt")

        # image topics
        self.topic_large_img = rospy.get_param("~topic_large_img", "/yolo_node/yolo_large_frame")
        self.topic_small_img = rospy.get_param("~topic_small_img", "/yolo_node/yolo_small_frame")

        # array topics
        self.topic_large_arr = rospy.get_param("~topic_large_arr", "/yolo_node/yolo_large_array")
        self.topic_small_arr = rospy.get_param("~topic_small_arr", "/yolo_node/yolo_small_array")

        # SORT topics
        self.topic_sort_small_arr = "/yolo_node/sort_small_predictions"
        self.topic_sort_small_img = "/yolo_node/sort_small_frame"
        self.topic_sort_large_arr = "/yolo_node/sort_large_predictions"
        self.topic_sort_large_img = "/yolo_node/sort_large_frame"

        # Fusion topics
        self.topic_fused_arr = "/yolo_node/fused_array"
        self.topic_fused_frame = "/yolo_node/fused_frame"

        # Device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rospy.loginfo("[yolo_dual] device: %s", self.device)

        # Load YOLO experts
        self.model_large = YOLO(self.large_weights)
        self.model_small = YOLO(self.small_weights)

        # Init SORT trackers
        self.mot_tracker_small = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
        self.mot_tracker_large = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
        self.msg_mat = init_matrix_array_ros_msg()

        # Fusion state
        self.smooth_buffer = []
        self.img_center = (224, 224)  # assuming 448x448 input size

        # ROS I/O
        self.bridge = CvBridge()
        self.pub_large_img = rospy.Publisher(self.topic_large_img, Image, queue_size=1)
        self.pub_small_img = rospy.Publisher(self.topic_small_img, Image, queue_size=1)
        self.pub_large_arr = rospy.Publisher(self.topic_large_arr, Float32MultiArray, queue_size=1)
        self.pub_small_arr = rospy.Publisher(self.topic_small_arr, Float32MultiArray, queue_size=1)

        # SORT publishers
        self.pub_sort_small_pred = rospy.Publisher(self.topic_sort_small_arr, Float32MultiArray, queue_size=1)
        self.pub_sort_small_frame = rospy.Publisher(self.topic_sort_small_img, Image, queue_size=1)
        self.pub_sort_large_pred = rospy.Publisher(self.topic_sort_large_arr, Float32MultiArray, queue_size=1)
        self.pub_sort_large_frame = rospy.Publisher(self.topic_sort_large_img, Image, queue_size=1)

        # Fusion publishers
        self.pub_fused_pred = rospy.Publisher(self.topic_fused_arr, Float32MultiArray, queue_size=1)
        self.pub_fused_frame = rospy.Publisher(self.topic_fused_frame, Image, queue_size=1)

        # Subscribe to camera
        self.sub = rospy.Subscriber(self.camera_topic, Image, self.cb_image,
                                    queue_size=1, buff_size=2 ** 24)

    def fuse_tracks(self, trackers_small, trackers_large, smooth_len=5):
        """
        Simplified single-target fusion:
        Always select the detection (from small or large) closest to image center.
        """
        fused = np.empty((0, 5))
        choice = "NONE"
        cx_img, cy_img = self.img_center

        # Combine both experts’ outputs
        candidates = []
        for t in trackers_small:
            cx = (t[0] + t[2]) / 2
            cy = (t[1] + t[3]) / 2
            dist = abs(cx - cx_img) + abs(cy - cy_img)
            candidates.append((dist, "Far", t))
        for t in trackers_large:
            cx = (t[0] + t[2]) / 2
            cy = (t[1] + t[3]) / 2
            dist = abs(cx - cx_img) + abs(cy - cy_img)
            candidates.append((dist, "Near", t))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            _, choice, best = candidates[0]
            fused = np.array([best])

            # Smoothing
            self.smooth_buffer.append(fused[0])
            if len(self.smooth_buffer) > smooth_len:
                self.smooth_buffer.pop(0)
            fused = np.array([np.mean(self.smooth_buffer, axis=0).astype(np.int32)])

        return fused, choice

    def cb_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        # Run BOTH experts
        res_large = self.model_large.predict(source=frame, verbose=False)
        res_small = self.model_small.predict(source=frame, verbose=False)

        # Publish YOLO images + arrays
        self.pub_large_img.publish(self.bridge.cv2_to_imgmsg(
            annotate_result(res_large, frame, "YOLO Near-range"), encoding="rgb8"))
        self.pub_small_img.publish(self.bridge.cv2_to_imgmsg(
            annotate_result(res_small, frame, "YOLO Far-range"), encoding="rgb8"))
        self.pub_large_arr.publish(detections_to_multiarray(res_large))
        self.pub_small_arr.publish(detections_to_multiarray(res_small))

        # ---------------- SORT Small ----------------
        tgt_boxes_small = []
        if hasattr(res_small[0], "boxes") and res_small[0].boxes is not None:
            for b in res_small[0].boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    tgt_boxes_small.append([x1, y1, x2, y2, 1.0])
        trackers_small = self.mot_tracker_small.update(
            np.array(tgt_boxes_small) if tgt_boxes_small else np.empty((0, 5))
        )

        np_frame_small = frame.copy()
        cv2.putText(np_frame_small, "SORT Far-range", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if len(trackers_small) > 0:
            for d in trackers_small:
                d = d.astype(np.int32)
                track_id = int(d[4])
                cv2.rectangle(np_frame_small, (d[0], d[1]), (d[2], d[3]), (0,165,255), 2)
                cv2.putText(np_frame_small, str(track_id), (d[0], d[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        else:
            cv2.putText(np_frame_small, "No Far Tracks", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.pub_sort_small_frame.publish(
            self.bridge.cv2_to_imgmsg(np_frame_small, encoding="rgb8")
        )
        self.pub_sort_small_pred.publish(pack_multiarray_ros_msg(self.msg_mat, trackers_small))

        # ---------------- SORT Large ----------------
        tgt_boxes_large = []
        if hasattr(res_large[0], "boxes") and res_large[0].boxes is not None:
            for b in res_large[0].boxes:
                if int(b.cls[0]) == 0:
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    tgt_boxes_large.append([x1, y1, x2, y2, 1.0])
        trackers_large = self.mot_tracker_large.update(
            np.array(tgt_boxes_large) if tgt_boxes_large else np.empty((0, 5))
        )

        np_frame_large = frame.copy()
        cv2.putText(np_frame_large, "SORT Near-range", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if len(trackers_large) > 0:
            for d in trackers_large:
                d = d.astype(np.int32)
                track_id = int(d[4])
                cv2.rectangle(np_frame_large, (d[0], d[1]), (d[2], d[3]), (0,165,255), 2)
                cv2.putText(np_frame_large, str(track_id), (d[0], d[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        else:
            cv2.putText(np_frame_large, "No Near Tracks", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.pub_sort_large_frame.publish(
            self.bridge.cv2_to_imgmsg(np_frame_large, encoding="rgb8")
        )
        self.pub_sort_large_pred.publish(pack_multiarray_ros_msg(self.msg_mat, trackers_large))

        # --- Fusion ---
        fused, choice = self.fuse_tracks(trackers_small, trackers_large)

        fused_frame = frame.copy()
        if len(fused) > 0:
            for d in fused:
                d = d.astype(np.int32)
                cv2.rectangle(fused_frame, (d[0], d[1]), (d[2], d[3]), (0,165,255), 2)
                cv2.putText(fused_frame, f"Expert ID {int(d[4])}", (d[0], d[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        else:
            cv2.putText(fused_frame, "Expert: NONE", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.putText(fused_frame, f"Expert: {choice}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Publish fused outputs
        self.pub_fused_frame.publish(
            self.bridge.cv2_to_imgmsg(fused_frame, encoding="rgb8")
        )
        self.pub_fused_pred.publish(pack_multiarray_ros_msg(self.msg_mat, fused))


def main():
    rospy.init_node("yolo_dual_node", anonymous=False)
    node = YoloDualNode()
    rospy.loginfo("[yolo_dual] ready with SORT + single-target Fusion (Small + Large).")
    rospy.spin()


if __name__ == "__main__":
    main()
