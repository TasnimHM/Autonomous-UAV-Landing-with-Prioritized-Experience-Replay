#!/usr/bin/env python3
"""
per_logger.py — Prioritized Experience Replay transition logger.

Reward design:
  Per step:
    Base:  -(|error_x| + |error_y|) / 448.0   (small negative for misalignment)
    Bonus: +0.5 when |error_x| < 15 AND |error_y| < 15  (positive for alignment)

  Terminal (based on XY landing error — not just success/timeout):
    Clean success:                     +50.0
    Timed out but landed close (<15m): +25.0  (was aligning correctly, just slow)
    Timed out, mediocre (15-30m):       0.0   (neutral)
    Timed out and far (>30m):          -10.0  (real failure)

  Why smart terminal:
    Sometimes the UAV aligns and descends correctly but times out because
    alignment took longer (especially from offset positions). Punishing these
    with -10 gives contradictory training signal. Instead we reward based on
    actual XY distance from helipad center at episode end.

Saves two CSV files:
  1. per_transitions.csv  — one row per control step (RL training)
  2. simulation_results.csv — one row per episode (paper plots)
"""

import csv
import os
import stat
import time
import numpy as np
import rospy

from std_msgs.msg import Float32MultiArray, Bool, Empty, Int32
from geometry_msgs.msg import Vector3, PoseStamped, Pose


# ── Save directory ─────────────────────────────────────────────────────────────
SAVE_DIR = "/catkin_ws/runs/per_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Output file paths ──────────────────────────────────────────────────────────
TRANSITIONS_CSV = os.path.join(SAVE_DIR, "per_transitions.csv")
SIM_RESULTS_CSV = os.path.join(SAVE_DIR, "simulation_results.csv")

# ── Image center reference ─────────────────────────────────────────────────────
CTR_X = 224.0
CTR_Y = 224.0

# ── Helipad ground truth center (XY only) ─────────────────────────────────────
HELIPAD_X = -80.0
HELIPAD_Y =  75.0

# ── Reward parameters ─────────────────────────────────────────────────────────
REWARD_SCALE        = 448.0   # larger → smaller negative per step
ALIGNMENT_BONUS     =   0.5   # positive reward when aligned
ALIGNMENT_THRESH    =  15.0   # pixels — matches controller TOL_X/Y

# Terminal rewards based on XY landing error
TERMINAL_SUCCESS    =  +50.0  # clean landing
TERMINAL_CLOSE      =  +25.0  # timed out but XY error < 15m (was doing well)
TERMINAL_MEDIOCRE   =    0.0  # timed out, XY error 15-30m (neutral)
TERMINAL_FAILURE    =  -10.0  # timed out AND far from helipad (real failure)

# Landing error thresholds for smart terminal
CLOSE_THRESHOLD     =  15.0   # metres — considered a good landing
MEDIOCRE_THRESHOLD  =  30.0   # metres — considered mediocre

# ── Hard transition threshold ──────────────────────────────────────────────────
HIGH_ERROR_THRESHOLD = 50.0   # pixels

# ── CSV headers ───────────────────────────────────────────────────────────────
TRANSITION_HEADER = [
    'episode_id',
    's_cx_large', 's_cy_large', 's_w_large', 's_h_large',
    's_cx_small', 's_cy_small', 's_w_small', 's_h_small',
    's_error_x', 's_error_y', 's_size', 's_altitude',
    'action', 'reward',
    'ns_cx_large', 'ns_cy_large', 'ns_w_large', 'ns_h_large',
    'ns_cx_small', 'ns_cy_small', 'ns_w_small', 'ns_h_small',
    'ns_error_x', 'ns_error_y', 'ns_size', 'ns_altitude',
    'done', 'flickering', 'expert_switched', 'high_error', 'is_hard', 'timed_out'
]

SIM_RESULTS_HEADER = [
    'episode_id',
    'x_i', 'y_i', 'z_i',
    'x_final', 'y_final', 'z_final',
    'time_to_land', 'success',
    'total_reward', 'n_transitions', 'n_hard', 'landing_error',
    'terminal_reward',
    'final_roll', 'final_pitch', 'final_yaw'
]


def parse_array_msg(msg):
    """Safely parse Float32MultiArray bbox. Returns None if empty."""
    if len(msg.layout.dim) == 0:
        return None
    if msg.layout.dim[0].size == 0:
        return None
    try:
        data = np.array(msg.data).reshape(
            msg.layout.dim[0].size,
            msg.layout.dim[1].size
        )
        return data[0]
    except Exception:
        return None


def fix_permissions(filepath):
    """Make file readable/writable by all — prevents lock icon."""
    try:
        os.chmod(filepath,
                 stat.S_IRUSR | stat.S_IWUSR |
                 stat.S_IRGRP | stat.S_IWGRP |
                 stat.S_IROTH | stat.S_IWOTH)
    except Exception as e:
        rospy.logwarn("📦 PER Logger: could not set permissions: %s", e)


def init_csv(filepath, header):
    """Create CSV with header if it does not already exist."""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
        fix_permissions(filepath)
        rospy.loginfo("📦 PER Logger: created %s", filepath)


class PERLogger:

    def __init__(self):
        rospy.init_node("per_logger")
        rospy.loginfo("📦 PER Logger started — saving to %s", SAVE_DIR)
        rospy.loginfo(
            "📦 Reward design: step=[align_bonus-error] | "
            "success=+%.0f | close=+%.0f | mediocre=%.0f | fail=%.0f",
            TERMINAL_SUCCESS, TERMINAL_CLOSE,
            TERMINAL_MEDIOCRE, TERMINAL_FAILURE
        )

        init_csv(TRANSITIONS_CSV, TRANSITION_HEADER)
        init_csv(SIM_RESULTS_CSV, SIM_RESULTS_HEADER)

        # ── Latest sensor data ─────────────────────────────────────────────────
        self.large_bbox  = None
        self.small_bbox  = None
        self.fused_bbox  = None
        self.tgt_box     = None
        self.altitude    = None
        self.guam_pose   = None
        self.episode_id  = 0

        # ── Episode state ──────────────────────────────────────────────────────
        self.transitions        = []
        self.prev_state         = None
        self.prev_action        = None
        self.prev_expert_id     = None
        self.episode_done       = False
        self.landed_success     = False
        self.episode_start_time = time.time()
        self.total_reward       = 0.0
        self.initial_x          = None
        self.initial_y          = None
        self.initial_z          = None

        # ── Subscribers ────────────────────────────────────────────────────────
        rospy.Subscriber("/yolo_node/yolo_large_array",
                         Float32MultiArray, self._cb_large)
        rospy.Subscriber("/yolo_node/yolo_small_array",
                         Float32MultiArray, self._cb_small)
        rospy.Subscriber("/yolo_node/fused_array",
                         Float32MultiArray, self._cb_fused)
        rospy.Subscriber("/controller_node/tgt_box_rcvd",
                         Vector3, self._cb_tgt_box)
        rospy.Subscriber("/jaxguam/pose",
                         PoseStamped, self._cb_pose)
        rospy.Subscriber("/episode/id",
                         Int32, self._cb_episode_id)
        rospy.Subscriber("/episode/initial_pose",
                         Pose, self._cb_initial_pose)
        rospy.Subscriber("/episode/done",
                         Bool, self._cb_episode_done)
        rospy.Subscriber("/episode/reset",
                         Empty, self._cb_episode_reset)

        rospy.loginfo("📦 PER Logger subscribed to all topics")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _cb_large(self, msg):
        self.large_bbox = parse_array_msg(msg)

    def _cb_small(self, msg):
        self.small_bbox = parse_array_msg(msg)

    def _cb_fused(self, msg):
        self.fused_bbox = parse_array_msg(msg)

    def _cb_tgt_box(self, msg):
        self.tgt_box = msg

    def _cb_pose(self, msg):
        self.altitude  = msg.pose.position.z
        self.guam_pose = msg

    def _cb_episode_id(self, msg):
        self.episode_id = msg.data

    def _cb_initial_pose(self, msg):
        self.initial_x          = msg.position.x
        self.initial_y          = msg.position.y
        self.initial_z          = msg.position.z
        self.episode_start_time = time.time()
        rospy.loginfo(
            "📦 PER Logger: ep %d start → x=%.1f y=%.1f z=%.1f",
            self.episode_id,
            self.initial_x, self.initial_y, self.initial_z
        )

    def _cb_episode_done(self, msg):
        if msg.data:
            self.episode_done   = True
            self.landed_success = True

    def _cb_episode_reset(self, msg):
        self._save_episode(timed_out=not self.landed_success)
        self._reset_episode_buffer()

    # ── State / action / reward ────────────────────────────────────────────────

    def _build_state(self):
        """Build 12-value state vector. Returns None if data not ready."""
        if self.tgt_box is None or self.altitude is None:
            return None

        if self.large_bbox is not None:
            x1, y1, x2, y2, _ = self.large_bbox
            cx_l = (x1 + x2) / 2.0
            cy_l = (y1 + y2) / 2.0
            w_l  = x2 - x1
            h_l  = y2 - y1
        else:
            cx_l = cy_l = w_l = h_l = 0.0

        if self.small_bbox is not None:
            x1, y1, x2, y2, _ = self.small_bbox
            cx_s = (x1 + x2) / 2.0
            cy_s = (y1 + y2) / 2.0
            w_s  = x2 - x1
            h_s  = y2 - y1
        else:
            cx_s = cy_s = w_s = h_s = 0.0

        error_x = self.tgt_box.x - CTR_X
        error_y = self.tgt_box.y - CTR_Y
        size    = self.tgt_box.z

        return np.array([
            cx_l, cy_l, w_l, h_l,
            cx_s, cy_s, w_s, h_s,
            error_x, error_y, size,
            self.altitude
        ], dtype=np.float32)

    def _get_action(self):
        """0=far expert, 1=near expert, -1=no detection."""
        if self.large_bbox is None and self.small_bbox is None:
            return -1

        def l1(bbox):
            if bbox is None:
                return float('inf')
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            return abs(cx - CTR_X) + abs(cy - CTR_Y)

        return 0 if l1(self.large_bbox) <= l1(self.small_bbox) else 1

    def _compute_reward(self, state):
        """
        Per-step reward:
          Base:  small negative for misalignment
          Bonus: positive when centered within threshold
        """
        error_x = state[8]
        error_y = state[9]

        step_reward = -(abs(error_x) + abs(error_y)) / REWARD_SCALE

        if abs(error_x) < ALIGNMENT_THRESH and abs(error_y) < ALIGNMENT_THRESH:
            step_reward += ALIGNMENT_BONUS

        return float(step_reward)

    def _compute_terminal_reward(self, timed_out):
        """
        Smart terminal reward based on XY landing error.
        Avoids punishing episodes where UAV was aligning correctly
        but ran out of time.
        For successful landings, reward scales with accuracy.
        """
        if not timed_out:
            landing_error = self._compute_landing_error()
            if landing_error < 3.0:
                rospy.loginfo("📦 PER Logger: excellent landing (%.2fm) → +50", landing_error)
                return +50.0    # excellent
            elif landing_error < 7.0:
                rospy.loginfo("📦 PER Logger: good landing (%.2fm) → +35", landing_error)
                return +35.0    # good
            elif landing_error < 12.0:
                rospy.loginfo("📦 PER Logger: okay landing (%.2fm) → +20", landing_error)
                return +20.0    # okay
            else:
                rospy.loginfo("📦 PER Logger: poor landing (%.2fm) → +10", landing_error)
                return +10.0    # completed but poor

        # For timeouts, check actual XY distance from helipad
        landing_error = self._compute_landing_error()

        if landing_error < CLOSE_THRESHOLD:
            # Was close to helipad — doing well, just needed more time
            rospy.loginfo(
                "📦 PER Logger: timeout but close (%.2fm) → partial success +%.0f",
                landing_error, TERMINAL_CLOSE
            )
            return TERMINAL_CLOSE
        elif landing_error < MEDIOCRE_THRESHOLD:
            # Mediocre — neutral signal
            rospy.loginfo(
                "📦 PER Logger: timeout mediocre (%.2fm) → neutral %.0f",
                landing_error, TERMINAL_MEDIOCRE
            )
            return TERMINAL_MEDIOCRE
        else:
            # Far from helipad — real failure
            rospy.loginfo(
                "📦 PER Logger: timeout far (%.2fm) → failure %.0f",
                landing_error, TERMINAL_FAILURE
            )
            return TERMINAL_FAILURE

    def _is_flickering(self):
        return self.large_bbox is None and self.small_bbox is None

    def _compute_landing_error(self):
        """XY euclidean distance from final position to helipad center."""
        if self.guam_pose is None:
            return -1.0
        dx = self.guam_pose.pose.position.x - HELIPAD_X
        dy = self.guam_pose.pose.position.y - HELIPAD_Y
        return float(np.sqrt(dx**2 + dy**2))

    # ── Save helpers ───────────────────────────────────────────────────────────

    def _save_transitions_to_csv(self, transitions):
        with open(TRANSITIONS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            for t in transitions:
                s  = t['state']
                ns = t['next_state']
                row = (
                    [t['episode_id']] +
                    [round(v, 4) for v in s.tolist()] +
                    [t['action'], round(t['reward'], 6)] +
                    [round(v, 4) for v in ns.tolist()] +
                    [
                        int(t['done']),
                        int(t['flickering']),
                        int(t['expert_switched']),
                        int(t['high_error']),
                        int(t['is_hard']),
                        int(t['timed_out']),
                    ]
                )
                writer.writerow(row)
        fix_permissions(TRANSITIONS_CSV)

    def _save_sim_result(self, timed_out, n_hard, terminal_reward):
        if self.initial_x is None or self.guam_pose is None:
            rospy.logwarn(
                "📦 PER Logger: missing pose for ep %d — skipping",
                self.episode_id
            )
            return

        elapsed       = time.time() - self.episode_start_time
        landing_error = self._compute_landing_error()

        # Result string based on smart terminal
        if not timed_out:
            result_str = "Success"
        elif terminal_reward >= TERMINAL_CLOSE:
            result_str = "PartialSuccess"
        elif terminal_reward == TERMINAL_MEDIOCRE:
            result_str = "Mediocre"
        else:
            result_str = "Fail"

        fx = self.guam_pose.pose.position.x
        fy = self.guam_pose.pose.position.y
        fz = self.guam_pose.pose.position.z

        try:
            from tf.transformations import euler_from_quaternion
            q = (
                self.guam_pose.pose.orientation.x,
                self.guam_pose.pose.orientation.y,
                self.guam_pose.pose.orientation.z,
                self.guam_pose.pose.orientation.w,
            )
            yaw, pitch, roll = euler_from_quaternion(q)
        except Exception:
            roll = pitch = yaw = 0.0

        with open(SIM_RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_id,
                round(self.initial_x, 4), round(self.initial_y, 4),
                round(self.initial_z, 4),
                round(fx, 4), round(fy, 4), round(fz, 4),
                round(elapsed, 4),
                result_str,
                round(self.total_reward, 4),
                len(self.transitions),
                n_hard,
                round(landing_error, 4),
                round(terminal_reward, 1),
                round(roll, 6), round(pitch, 6), round(yaw, 6)
            ])
        fix_permissions(SIM_RESULTS_CSV)

    def _save_episode(self, timed_out=False):
        if len(self.transitions) == 0:
            rospy.logwarn("📦 PER Logger: no transitions for ep %d",
                          self.episode_id)
            return

        # Smart terminal reward
        terminal = self._compute_terminal_reward(timed_out)

        self.transitions[-1]['reward']   += terminal
        self.transitions[-1]['done']      = True
        self.transitions[-1]['timed_out'] = timed_out
        self.total_reward                += terminal

        n_hard = sum(1 for t in self.transitions if t['is_hard'])

        self._save_transitions_to_csv(self.transitions)
        self._save_sim_result(timed_out, n_hard, terminal)

        landing_error = self._compute_landing_error()
        rospy.loginfo(
            "📦 PER Logger: ep %d → %d steps | %d hard | "
            "reward=%.1f | error=%.2fm | terminal=%.0f | %s",
            self.episode_id,
            len(self.transitions),
            n_hard,
            self.total_reward,
            landing_error,
            terminal,
            "✅ Success" if not timed_out else
            "⚠️ PartialSuccess" if terminal >= TERMINAL_CLOSE else
            "➖ Mediocre" if terminal == TERMINAL_MEDIOCRE else
            "❌ Fail"
        )

    def _reset_episode_buffer(self):
        self.transitions        = []
        self.prev_state         = None
        self.prev_action        = None
        self.prev_expert_id     = None
        self.episode_done       = False
        self.landed_success     = False
        self.episode_start_time = time.time()
        self.total_reward       = 0.0
        rospy.loginfo("📦 PER Logger: buffer cleared for next episode")

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():

            state = self._build_state()
            if state is None:
                rate.sleep()
                continue

            action          = self._get_action()
            flickering      = self._is_flickering()
            expert_switched = (
                self.prev_expert_id is not None and
                action != self.prev_expert_id and
                action != -1
            )
            high_error = (
                abs(state[8]) > HIGH_ERROR_THRESHOLD or
                abs(state[9]) > HIGH_ERROR_THRESHOLD
            )
            is_hard = flickering or expert_switched or high_error

            if self.prev_state is not None and self.prev_action is not None:
                reward = self._compute_reward(self.prev_state)
                self.total_reward += reward

                self.transitions.append({
                    'state':           self.prev_state,
                    'action':          self.prev_action,
                    'reward':          reward,
                    'next_state':      state,
                    'done':            False,
                    'episode_id':      self.episode_id,
                    'flickering':      flickering,
                    'expert_switched': expert_switched,
                    'high_error':      high_error,
                    'is_hard':         is_hard,
                    'timed_out':       False,
                })

            self.prev_state     = state
            self.prev_action    = action
            self.prev_expert_id = action if action != -1 else self.prev_expert_id

            rate.sleep()


if __name__ == "__main__":
    logger = PERLogger()
    logger.run()