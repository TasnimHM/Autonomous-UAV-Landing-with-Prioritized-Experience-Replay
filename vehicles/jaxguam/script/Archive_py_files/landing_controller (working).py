#!/usr/bin/env python3
import numpy as np
import rospy
#from std_msgs.msg import Float32MultiArray
from std_msgs.msg import Float32MultiArray, Bool, Empty, Int32
from geometry_msgs.msg import Twist, Vector3

# --- Globals ---
TRACKING_ARRAY_RECEIVED = None

# --- Episode lifecycle ---
EPISODE_ID = 0
EPISODE_RESET_FLAG = False
DONE_SENT = False

def cb_episode_id(msg):
    global EPISODE_ID
    EPISODE_ID = msg.data

def cb_episode_reset(msg):
    global EPISODE_RESET_FLAG
    EPISODE_RESET_FLAG = True

def fnc_callback(msg):
    """Receive YOLO detections (Large/Small/Fused)."""
    global TRACKING_ARRAY_RECEIVED
    TRACKING_ARRAY_RECEIVED = msg


# PID gains (tuned for smoother response)
P_gain = 0.03
D_gain = 0.0015
I_gain = 0.0
FREQ_LOW_LEVEL = 10  # Hz

if __name__ == "__main__":
    rospy.init_node("controller_node")
    rospy.loginfo("🧠 Landing Controller: stable-centering + lock-hold + descent")

    sub = rospy.Subscriber("/yolo_node/fused_array", Float32MultiArray, fnc_callback)
    pub_tgt_box = rospy.Publisher("/controller_node/tgt_box_rcvd", Vector3, queue_size=10)
    pub_vel_cmd = rospy.Publisher("/controller_node/vel_cmd", Twist, queue_size=10)
    pub_episode_done = rospy.Publisher("/episode/done", Bool, queue_size=1)
    rospy.Subscriber("/episode/reset", Empty, cb_episode_reset)
    rospy.Subscriber("/episode/id", Int32, cb_episode_id)
    rate = rospy.Rate(FREQ_LOW_LEVEL)

    # Target values (image center)
    CTR_X_POS = 224
    CTR_Y_POS = 224
    AREA_SIZE = 60  # reference helipad size near ground

    # Thresholds and tolerances
    TOL_X = 15
    TOL_Y = 15
    DEADZONE = 8

    # States
    aligned_state = False
    lock_counter = 0
    prev_x_ctr = prev_y_ctr = 0.0
    last_detection = None
    last_detection_time = rospy.Time.now()
    vel_cmd_tracking = Twist()

    rospy.loginfo("Controller running — XY align ➜ lock ➜ descend ➜ hover...")

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        if EPISODE_RESET_FLAG:
            aligned_state = False
            lock_counter = 0
            prev_x_ctr = prev_y_ctr = 0.0
            last_detection = None
            last_detection_time = rospy.Time.now()
            vel_cmd_tracking = Twist()
            pub_vel_cmd.publish(vel_cmd_tracking)
            DONE_SENT = False

            rospy.loginfo(f"🔄 Episode reset applied (episode_id={EPISODE_ID})")
            EPISODE_RESET_FLAG = False

        if TRACKING_ARRAY_RECEIVED is not None:
            h = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            w = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((h, w))

            # --- Detection selection ---
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
                rospy.logwarn_throttle(2.0, "⚠️ Using memory detection (YOLO flicker)")
            else:
                vel_cmd_tracking.linear.x = vel_cmd_tracking.linear.y = vel_cmd_tracking.linear.z = 0.0
                pub_vel_cmd.publish(vel_cmd_tracking)
                rate.sleep()
                continue

            # --- Bounding box parameters ---
            x_ctr = (x1 + x2) / 2
            y_ctr = (y1 + y2) / 2
            size = (x2 - x1) * (y2 - y1) / 1000.0

            # --- Smoothing (EMA) ---
            x_ctr = 0.7 * x_ctr + 0.3 * prev_x_ctr
            y_ctr = 0.7 * y_ctr + 0.3 * prev_y_ctr
            prev_x_ctr, prev_y_ctr = x_ctr, y_ctr

            # --- Error computation ---
            error_x = x_ctr - CTR_X_POS
            error_y = y_ctr - CTR_Y_POS

            if abs(error_x) < DEADZONE: error_x = 0
            if abs(error_y) < DEADZONE: error_y = 0

            # --- PD control for XY ---
            dx = error_x * 0.1
            dy = error_y * 0.1
            cmd_vx = P_gain * error_x - D_gain * dx
            cmd_vy = P_gain * -error_y - D_gain * dy  # inverted image Y

            # --- Alignment detection with lock hold ---
            if not aligned_state:
                if abs(error_x) < TOL_X and abs(error_y) < TOL_Y:
                    lock_counter += 1
                    if lock_counter > 8:  # hold for ~0.8 sec at 10Hz
                        aligned_state = True
                        rospy.loginfo("✅ Aligned — locking XY and starting descent")
                else:
                    lock_counter = 0
            else:
                # Once aligned, ignore tiny jitter
                error_x = 0
                error_y = 0

            # --- Descent control ---
            if aligned_state:
                cmd_vz = -0.6 + 0.002 * (AREA_SIZE - size) #-0.6 to -0.9
                cmd_vz = np.clip(cmd_vz, -0.8, -0.25)

                # freeze XY during descent
                vel_cmd_tracking.linear.x *= 0.0
                vel_cmd_tracking.linear.y *= 0.0

                if size > (AREA_SIZE * 2.2):
                    cmd_vz = 0.0


                    if not DONE_SENT:
                        rospy.loginfo(f"🛬 Landing complete — hover (episode_id={EPISODE_ID})")
                        pub_episode_done.publish(True)
                        DONE_SENT = True
            else:
                cmd_vz = 0.0

            # --- Blend XY smoothly (damped & corrected mapping) ---
            vel_cmd_tracking.linear.x = 0.8 * vel_cmd_tracking.linear.x + 0.2 * (0.4 * cmd_vy)
            vel_cmd_tracking.linear.y = 0.8 * vel_cmd_tracking.linear.y + 0.2 * (0.4 * cmd_vx)
            vel_cmd_tracking.linear.z = cmd_vz

            # --- Safety limit ---
            vel_cmd_tracking.linear.x = np.clip(vel_cmd_tracking.linear.x, -1.5, 1.5)
            vel_cmd_tracking.linear.y = np.clip(vel_cmd_tracking.linear.y, -1.5, 1.5)
            vel_cmd_tracking.linear.z = np.clip(vel_cmd_tracking.linear.z, -1.5, 1.5)

            # --- Debug info ---
            pub_tgt_box.publish(Vector3(x_ctr, y_ctr, size))
            rospy.loginfo_throttle(
                1.0,
                f"[CTRL] err=({error_x:.1f},{error_y:.1f}) | "
                f"vx={cmd_vx:.2f}, vy={cmd_vy:.2f}, vz={cmd_vz:.2f} | "
                f"size={size:.1f} | aligned={aligned_state} | lock={lock_counter}"
            )

        pub_vel_cmd.publish(vel_cmd_tracking)
        rate.sleep()
