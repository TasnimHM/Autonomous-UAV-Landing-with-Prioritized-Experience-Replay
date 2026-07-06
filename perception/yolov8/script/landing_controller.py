#!/usr/bin/env python3
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist, Vector3

TRACKING_ARRAY_RECEIVED = None

def fnc_callback(msg):
    global TRACKING_ARRAY_RECEIVED
    TRACKING_ARRAY_RECEIVED = msg

P_gain = 0.045
D_gain = 0.002
FREQ_LOW_LEVEL = 10

if __name__ == '__main__':
    rospy.init_node('controller_node')
    rospy.loginfo("🧠 Landing Controller (align once → lock XY → descend)")

    sub = rospy.Subscriber('/yolo_node/fused_array', Float32MultiArray, fnc_callback)
    pub_tgt_box = rospy.Publisher('/controller_node/tgt_box_rcvd', Vector3, queue_size=10)
    pub_vel_cmd = rospy.Publisher('/controller_node/vel_cmd', Twist, queue_size=10)

    rate = rospy.Rate(FREQ_LOW_LEVEL)

    CTR_X_POS = 224
    CTR_Y_POS = 224
    AREA_SIZE = 60
    TOL_X = 25
    TOL_Y = 25
    DEADZONE = 10

    vel_cmd_tracking = Twist()
    aligned_state = False
    xy_locked = False

    prev_x_ctr = prev_y_ctr = 0.0
    last_detection_time = rospy.Time.now()
    last_detection = None

    rospy.loginfo("Controller running — will lock XY once centered and descend straight...")

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if TRACKING_ARRAY_RECEIVED is not None:
            h = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            w = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((h, w))

            # Detection
            if len(np_tracking) > 0:
                centers = []
                for (x1, y1, x2, y2, track_id) in np_tracking:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    dist = abs(cx - CTR_X_POS) + abs(cy - CTR_Y_POS)
                    centers.append((dist, (x1, y1, x2, y2, track_id)))
                _, the_obj = min(centers, key=lambda x: x[0])
                x1, y1, x2, y2, track_id = the_obj
                last_detection = (x1, y1, x2, y2)
                last_detection_time = now

            elif (now - last_detection_time).to_sec() < 2.5 and last_detection is not None:
                x1, y1, x2, y2 = last_detection
            else:
                vel_cmd_tracking.linear.x = vel_cmd_tracking.linear.y = vel_cmd_tracking.linear.z = 0.0
                pub_vel_cmd.publish(vel_cmd_tracking)
                rate.sleep()
                continue

            # Compute center
            x_ctr = (x1 + x2) / 2
            y_ctr = (y1 + y2) / 2
            size = (x2 - x1) * (y2 - y1) / 1000.0

            # Smoothing
            x_ctr = 0.8 * x_ctr + 0.2 * prev_x_ctr
            y_ctr = 0.8 * y_ctr + 0.2 * prev_y_ctr
            dx = x_ctr - prev_x_ctr
            dy = y_ctr - prev_y_ctr
            prev_x_ctr, prev_y_ctr = x_ctr, y_ctr

            error_x = (x_ctr - CTR_X_POS)
            error_y = (y_ctr - CTR_Y_POS)

            if abs(error_x) < DEADZONE: error_x = 0.0
            if abs(error_y) < DEADZONE: error_y = 0.0

            # --- Stage 1: Align until centered ---
            if not aligned_state and abs(error_x) < TOL_X and abs(error_y) < TOL_Y:
                aligned_state = True
                rospy.loginfo("✅ Centered once — locking XY and starting descent")

            # --- Once aligned, lock XY completely ---
            if aligned_state and not xy_locked:
                xy_locked = True
                lock_size = size
                rospy.loginfo(f"🔒 XY locked at size={lock_size:.1f}, beginning descent...")

            # --- Control logic ---
            if not xy_locked:
                # still aligning
                cmd_vx = P_gain * error_x - D_gain * dx
                cmd_vy = P_gain * -error_y - D_gain * dy
                cmd_vz = -0.6  # normal descent
            else:
                # XY locked: go straight down until close enough
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_vz = -0.8 if size < (AREA_SIZE * 2.0) else 0.0

            cmd_vx = np.clip(cmd_vx, -1.0, 1.0)
            cmd_vy = np.clip(cmd_vy, -1.0, 1.0)

            vel_cmd_tracking.linear.x = cmd_vy
            vel_cmd_tracking.linear.y = -cmd_vx
            vel_cmd_tracking.linear.z = cmd_vz

            pub_tgt_box.publish(Vector3(x_ctr, y_ctr, size))
            pub_vel_cmd.publish(vel_cmd_tracking)

            rospy.loginfo_throttle(
                1.0,
                f"[CTRL] err=({error_x:.1f},{error_y:.1f}) | xy_locked={xy_locked} | "
                f"vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, vz={cmd_vz:.2f} | size={size:.1f}"
            )

        rate.sleep()
