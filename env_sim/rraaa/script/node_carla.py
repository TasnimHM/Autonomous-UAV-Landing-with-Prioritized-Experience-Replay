#!/usr/bin/env python3
"""
node_carla.py — CARLA environment node with pose-aware reset support.

Changes from original:
  • Subscribes to /episode/initial_pose (geometry_msgs/Pose).
  • Stores the latest pose so that environment.reset(pose) can teleport the
    CARLA actor to the randomised starting location.
  • environment.reset() must accept an optional 'pose' kwarg — see notes below.
"""

import argparse
import atexit
import carla
import glob
import numpy as np
import os
import rospy
import signal
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from tools.environment import Environment
from geometry_msgs.msg import Pose

from loguru import logger as log
sys.path.append(os.path.abspath('/catkin_ws/src/env_sim/utils'))
from utils.config import load_yaml_file
from utils import constants

FREQ_LOW_LEVEL = 10


class GracefulShutdown:
    def __init__(self, environment):
        self.environment = environment
        atexit.register(self.shutdown)
        signal.signal(signal.SIGINT,  self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        self.environment.destroy()
        log.info("Carla environment destroyed.")


def run_carla_node(args, client):
    pygame.init()
    config = load_yaml_file(constants.merged_config_path, __file__)

    rospy.init_node('carla_node')
    rospy.set_param('tracking_control', False)

    environment = Environment(args, client, config)
    _ = GracefulShutdown(environment)

    rospy.set_param('reset_called',  False)
    rospy.set_param('episode_done',  False)
    rospy.set_param('done_ack',      False)

    # ── Store the latest requested initial pose ───────────────────────────────
    # Written by the subscriber below; read in the main loop when reset fires.
    latest_initial_pose = {'pose': None}   # mutable container for closure

    def _initial_pose_callback(msg: Pose):
        latest_initial_pose['pose'] = msg

    rospy.Subscriber('/episode/initial_pose', Pose, _initial_pose_callback)

    try:
        rate = rospy.Rate(FREQ_LOW_LEVEL)

        while not rospy.is_shutdown():
            environment.client_clock.tick_busy_loop(FREQ_LOW_LEVEL)
            environment.tick()
            rate.sleep()

            # ── Episode-done reset (from high-level decision maker) ───────────
            if not rospy.get_param('done_ack') and rospy.get_param('episode_done'):
                _do_reset(environment, latest_initial_pose['pose'])
                rospy.set_param('done_ack', True)

            elif rospy.get_param('done_ack') and not rospy.get_param('episode_done'):
                rospy.set_param('done_ack', False)

            # ── Manual / episode-manager reset (reset_called param) ───────────
            if rospy.get_param('reset_called'):
                log.info("[carla_node] reset_called detected — resetting environment.")
                _do_reset(environment, latest_initial_pose['pose'])
                rospy.set_param('reset_called', False)
                rospy.set_param('reset_ack',    True)
            else:
                rospy.set_param('reset_ack', False)

    finally:
        environment.destroy()


def _do_reset(environment: "Environment", pose):
    """
    Call environment.reset() with the pose if the Environment class supports it,
    otherwise fall back to the no-arg version.

    HOW TO ADD POSE SUPPORT TO environment.reset():
    ------------------------------------------------
    In tools/environment.py, change the signature to:

        def reset(self, pose=None):
            if pose is not None:
                # Teleport the ego actor
                transform = carla.Transform(
                    carla.Location(
                        x=pose.position.x,
                        y=pose.position.y,
                        z=pose.position.z,
                    ),
                    carla.Rotation(pitch=0, yaw=0, roll=0),
                )
                self.ego_vehicle.set_transform(transform)
            # ... rest of your existing reset logic ...

    The set_transform() call is the CARLA Python API method and is safe to call
    on an already-spawned actor — it does NOT re-spawn (which is what was
    causing instability before).
    """
    try:
        import inspect
        sig = inspect.signature(environment.reset)
        if 'pose' in sig.parameters and pose is not None:
            environment.reset(pose=pose)
            log.info(
                "[carla_node] reset with pose → x={:.2f} y={:.2f} z={:.2f}",
                pose.position.x, pose.position.y, pose.position.z,
            )
        else:
            environment.reset()
            if pose is None:
                log.warning("[carla_node] reset called but no pose received yet — "
                            "using default CARLA position.")
    except Exception as e:
        log.error(f"[carla_node] reset failed: {e}")
        environment.reset()   # safe fallback


def main():
    argparser = argparse.ArgumentParser(description='ROS CARLA NODE')
    argparser.add_argument('--host',      metavar='H', default='127.0.0.1')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int)
    argparser.add_argument('--res',       metavar='WIDTHxHEIGHT', default='800x400')
    argparser.add_argument('--asynch',    action='store_false')

    # Traffic settings
    argparser.add_argument('-n', '--number-of-vehicles',  metavar='N', default=30,  type=int)
    argparser.add_argument('-w', '--number-of-walkers',   metavar='W', default=0,   type=int)
    argparser.add_argument('--safe',          action='store_true')
    argparser.add_argument('--filterv',       metavar='PATTERN', default='vehicle.*')
    argparser.add_argument('--generationv',   metavar='G',       default='All')
    argparser.add_argument('--filterw',       metavar='PATTERN', default='walker.pedestrian.*')
    argparser.add_argument('--generationw',   metavar='G',       default='2')
    argparser.add_argument('--tm-port',       metavar='P',       default=8000, type=int)
    argparser.add_argument('--hybrid',        action='store_true')
    argparser.add_argument('-s', '--seed',    metavar='S',       type=int)
    argparser.add_argument('--seedw',         metavar='S',       default=0,    type=int)
    argparser.add_argument('--car-lights-on', action='store_true', default=False)
    argparser.add_argument('--hero',          action='store_true', default=False)
    argparser.add_argument('--respawn',       action='store_true', default=False)
    argparser.add_argument('--no-rendering',  action='store_true', default=False)

    args, unknown   = argparser.parse_known_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    args.asynch = False

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        run_carla_node(args, client)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass