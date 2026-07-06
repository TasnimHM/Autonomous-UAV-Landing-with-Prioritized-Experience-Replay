#!/usr/bin/env python3
"""
landing_controller.py — Landing controller for RL data collection.

Current version:
  1. Adaptive P gain — scales down for large errors to prevent overshoot
     when UAV spawns far from helipad.
  2. lock_counter > 8 — prevents false alignment at random start positions
  3. TOL_X/Y = 10 — tighter alignment tolerance
  4. Once aligned, stays aligned — removed re-alignment check during descent
     because it caused infinite re-align loops when YOLO detection shifted.
  5. XY frozen during descent — prevents GUAM Z attenuation
  6. Descent speed scales with size — fast high, slow near ground
"""

import numpy as np
import rospy
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


# ── PID gains ─────────────────────────────────────────────────────────────────
P_gain = 0.05    # base P gain — scaled down adaptively for large errors
D_gain = 0.003
I_gain = 0.0
FREQ_LOW_LEVEL = 10  # Hz


if __name__ == "__main__":
    rospy.init_node("controller_node")
    rospy.loginfo("🧠 Landing Controller: adaptive gain + scaled descent")

    sub              = rospy.Subscriber("/yolo_node/fused_array", Float32MultiArray, fnc_callback)
    pub_tgt_box      = rospy.Publisher("/controller_node/tgt_box_rcvd", Vector3, queue_size=10)
    pub_vel_cmd      = rospy.Publisher("/controller_node/vel_cmd", Twist, queue_size=10)
    pub_episode_done = rospy.Publisher("/episode/done", Bool, queue_size=1)
    rospy.Subscriber("/episode/reset", Empty, cb_episode_reset)
    rospy.Subscriber("/episode/id",    Int32, cb_episode_id)
    rate = rospy.Rate(FREQ_LOW_LEVEL)

    # ── Image-space targets ────────────────────────────────────────────────────
    CTR_X_POS = 224
    CTR_Y_POS = 224

    # ── Helipad size thresholds ────────────────────────────────────────────────
    AREA_SIZE       = 60
    AREA_SIZE_CLOSE = 65   # trigger hover

    # ── Alignment thresholds ──────────────────────────────────────────────────
    TOL_X    = 10
    TOL_Y    = 10
    DEADZONE = 8

    # ── Descent speed parameters ──────────────────────────────────────────────
    VZ_MAX   = -1.0
    VZ_MIN   = -0.25
    VZ_SCALE =  0.012

    # ── State variables ───────────────────────────────────────────────────────
    aligned_state       = False
    lock_counter        = 0
    prev_x_ctr          = 0.0
    prev_y_ctr          = 0.0
    last_detection      = None
    last_detection_time = rospy.Time.now()
    vel_cmd_tracking    = Twist()

    rospy.loginfo("Controller running — XY align ➜ lock ➜ scaled descent ➜ hover...")

    while not rospy.is_shutdown():
        now = rospy.Time.now()

        # ── Episode reset ──────────────────────────────────────────────────────
        if EPISODE_RESET_FLAG:
            aligned_state       = False
            lock_counter        = 0
            prev_x_ctr          = 0.0
            prev_y_ctr          = 0.0
            last_detection      = None
            last_detection_time = rospy.Time.now()
            vel_cmd_tracking    = Twist()
            pub_vel_cmd.publish(vel_cmd_tracking)
            DONE_SENT = False
            rospy.loginfo(f"🔄 Episode reset applied (episode_id={EPISODE_ID})")
            EPISODE_RESET_FLAG  = False

        if TRACKING_ARRAY_RECEIVED is not None:
            h = TRACKING_ARRAY_RECEIVED.layout.dim[0].size
            w = TRACKING_ARRAY_RECEIVED.layout.dim[1].size
            np_tracking = np.array(TRACKING_ARRAY_RECEIVED.data).reshape((h, w))

            # ── Detection selection ────────────────────────────────────────────
            if len(np_tracking) > 0:
                centers = []
                for (x1, y1, x2, y2, track_id) in np_tracking:
                    cx   = (x1 + x2) / 2
                    cy   = (y1 + y2) / 2
                    dist = abs(cx - CTR_X_POS) + abs(cy - CTR_Y_POS)
                    centers.append((dist, (x1, y1, x2, y2, track_id)))
                _, the_obj = min(centers, key=lambda x: x[0])
                x1, y1, x2, y2, track_id = the_obj
                last_detection      = (x1, y1, x2, y2)
                last_detection_time = now

            elif (now - last_detection_time).to_sec() < 2.5 and last_detection is not None:
                x1, y1, x2, y2 = last_detection
                rospy.logwarn_throttle(2.0, "⚠️ Using memory detection (YOLO flicker)")

            else:
                rospy.logwarn_throttle(2.0, "⚠️ No detection — slowly descending to search")
                vel_cmd_tracking.linear.x = 0.0
                vel_cmd_tracking.linear.y = 0.0
                vel_cmd_tracking.linear.z = -0.3
                pub_vel_cmd.publish(vel_cmd_tracking)
                rate.sleep()
                continue

            # ── Bounding box parameters ────────────────────────────────────────
            x_ctr = (x1 + x2) / 2
            y_ctr = (y1 + y2) / 2
            size  = (x2 - x1) * (y2 - y1) / 1000.0

            # ── Smoothing (EMA) ────────────────────────────────────────────────
            x_ctr      = 0.7 * x_ctr + 0.3 * prev_x_ctr
            y_ctr      = 0.7 * y_ctr + 0.3 * prev_y_ctr
            prev_x_ctr = x_ctr
            prev_y_ctr = y_ctr

            # ── Error computation ──────────────────────────────────────────────
            error_x = x_ctr - CTR_X_POS
            error_y = y_ctr - CTR_Y_POS

            if abs(error_x) < DEADZONE: error_x = 0
            if abs(error_y) < DEADZONE: error_y = 0

            # ── Adaptive P gain ────────────────────────────────────────────────
            # Reduces P gain for large errors to prevent overshoot on approach.
            # At error=10px:  adaptive_P = 0.05 / (1 + 0.02*10)  = 0.042
            # At error=100px: adaptive_P = 0.05 / (1 + 0.02*100) = 0.017
            # At error=200px: adaptive_P = 0.05 / (1 + 0.02*200) = 0.010
            error_magnitude = np.sqrt(error_x**2 + error_y**2)
            adaptive_P      = P_gain / (1.0 + 0.02 * error_magnitude)

            # ── PD control for XY ──────────────────────────────────────────────
            dx     = error_x * 0.1
            dy     = error_y * 0.1
            cmd_vx = adaptive_P * error_x  - D_gain * dx
            cmd_vy = adaptive_P * -error_y - D_gain * dy   # inverted image Y

            # ── Alignment detection with lock hold ─────────────────────────────
            # lock_counter > 8 prevents false alignment at random start positions.
            # Once aligned, stays aligned until episode resets — removing the
            # re-alignment check during descent because it caused infinite loops
            # when YOLO detection shifted slightly mid-descent.
            if not aligned_state:
                if abs(error_x) < TOL_X and abs(error_y) < TOL_Y:
                    lock_counter += 1
                    if lock_counter > 8:
                        aligned_state = True
                        rospy.loginfo("✅ Aligned — starting descent")
                else:
                    lock_counter = 0
            # Note: no else block here — once aligned, stays aligned

            # ── Descent control ────────────────────────────────────────────────
            if aligned_state:

                # Scale descent speed with helipad size
                cmd_vz = VZ_MAX + VZ_SCALE * size
                cmd_vz = np.clip(cmd_vz, VZ_MAX, VZ_MIN)

                # Freeze XY during descent — prevents GUAM Z attenuation
                vel_cmd_tracking.linear.x = 0.0
                vel_cmd_tracking.linear.y = 0.0

                # Landing complete
                if size > AREA_SIZE_CLOSE:
                    cmd_vz = 0.0
                    if not DONE_SENT:
                        rospy.loginfo(f"🛬 Landing complete — hover (episode_id={EPISODE_ID})")
                        pub_episode_done.publish(True)
                        DONE_SENT = True

            else:
                cmd_vz = 0.0

                # XY alignment phase
                vel_cmd_tracking.linear.x = (
                    0.7 * vel_cmd_tracking.linear.x + 0.3 * (0.4 * cmd_vy)
                )
                vel_cmd_tracking.linear.y = (
                    0.7 * vel_cmd_tracking.linear.y + 0.3 * (0.4 * cmd_vx)
                )

            vel_cmd_tracking.linear.z = cmd_vz

            # ── Safety limits ──────────────────────────────────────────────────
            vel_cmd_tracking.linear.x = np.clip(vel_cmd_tracking.linear.x, -2.0, 2.0)
            vel_cmd_tracking.linear.y = np.clip(vel_cmd_tracking.linear.y, -2.0, 2.0)
            vel_cmd_tracking.linear.z = np.clip(vel_cmd_tracking.linear.z, -2.0, 0.5)

            # ── Debug ──────────────────────────────────────────────────────────
            pub_tgt_box.publish(Vector3(x_ctr, y_ctr, size))
            rospy.loginfo_throttle(
                1.0,
                f"[CTRL] err=({error_x:.1f},{error_y:.1f}) | "
                f"adaptP={adaptive_P:.4f} | "
                f"vx={vel_cmd_tracking.linear.x:.2f}, "
                f"vy={vel_cmd_tracking.linear.y:.2f}, "
                f"vz={cmd_vz:.2f} | "
                f"size={size:.1f} | aligned={aligned_state} | lock={lock_counter}"
            )

        pub_vel_cmd.publish(vel_cmd_tracking)
        rate.sleep()
    