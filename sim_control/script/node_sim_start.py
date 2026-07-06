#!/usr/bin/env python3

import rospy
from rospy.msg import AnyMsg
from std_msgs.msg import Bool

from utils.config import load_yaml_file, write_shared_tmp_file, log
from utils import constants

# class SimStart:
#     def __init__(self):
#         self.config = load_yaml_file(constants.merged_config_path, __file__)

#         rospy.init_node('sim_start')
#         self.rate = rospy.Rate(constants.frequency_low)

#         self.sim_start_pub = rospy.Publisher('/sim_start/started', Bool, queue_size=1)
#         self.sim_start_pub.publish(False)

#         rospy.Subscriber(f"/{self.config['ego_vehicle']['type']}/pose", AnyMsg, self.callback)

#     def callback(self, msg):
#         if msg:
#             log.trace("Simulator has started.")
#             self.sim_start_pub.publish(True)
#         else:
#             log.trace("Simulator not started.")

#     def run(self):
#         while not rospy.is_shutdown():
#             self.rate.sleep()


# if __name__ == "__main__":
#     simstart = SimStart()
#     simstart.run()
