#!/usr/bin/env python

# import queue
import pdb
import rospy
import scipy
import scipy.stats
import math
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from recycling_stretch.msg import ArucoRefMarker, WristExtension, CalibrationParams
from recycling_stretch.srv import PushObject, PushObjectRequest, PushObjectResponse, \
                                PullObject, PullObjectRequest, PullObjectResponse

SLEEP_RATE = 10
VELOCITY_MARKER_ID = 'velocity_marker'
WRIST_MARKER_ID = 'top_wrist_marker'

lift_calibration_param = CalibrationParams()
wrist_calibration_param = CalibrationParams()

class VelocityCalculator(object):
    '''
    Object for tuning calibration parameters for conveyor belt velocity and robot extension speeds
    '''
    def __init__(self):
        self.aruco_marker_position_buffer = []
        self.velocity = 0
        self.default_wrist_position = 0
        self.default_lift_position = 0
        self.wrist_extension_values, self.wrist_extension_times, self.lift_position_values, self.lift_position_times = [], [], [], []
        self.wrist_calibration, self.lift_calibration = {'slope': 0, 'intercept': 0}, {'slope': 0, 'intercept': 0}

        # Publisher
        self.aruco_velocity_pub = rospy.Publisher('/recycling_stretch/belt_velocity', Float32, queue_size=1)
        self.wrist_calibration_pub = rospy.Publisher('/recycling_stretch/wrist_calibration', CalibrationParams, queue_size=1)
        self.lift_calibration_pub = rospy.Publisher('/recycling_stretch/lift_calibration', CalibrationParams, queue_size=1)

        # Subscriber
        self.joint_states = rospy.wait_for_message('/stretch/joint_states', JointState)
        self.joint_states_cb(self.joint_states)

        self.aruco_marker_position_sub = rospy.Subscriber('/recycling_stretch/marker_position', ArucoRefMarker, self.aruco_marker_position_cb, queue_size=1)
        self.wrist_extension_sub = rospy.Subscriber('/recycling_stretch/wrist_extension_time', WristExtension, self.calibrate_extensions_cb, queue_size=1)
        self.lift_extension_sub = rospy.Subscriber('/recycling_stretch/lift_extension_time', WristExtension, self.calibrate_lift_extensions_cb, queue_size=1)
        
        # Service Proxies
        rospy.wait_for_service("/recycling_stretch/push_object_srv")
        self.push_object_sp = rospy.ServiceProxy("/recycling_stretch/push_object_srv", PushObject)

        rospy.wait_for_service("/recycling_stretch/pull_object_srv")
        self.pull_object_sp = rospy.ServiceProxy("/recycling_stretch/pull_object_srv", PullObject)
    
        
    def joint_states_cb(self, joint_msg):
        """
        Callback function that receives a joint_msg from the Stretch RE1 about the position of the wrist and lift motor 
        relative to the starting position

        Parameters:
                    joint_msg (msg) Message of type JointState with string[] name and float64[] position corresponding to each motor

        """
        dict = {"wrist_extension" : 0, "joint_lift" : 0}
        for i, joint in enumerate(joint_msg.name):
            if joint == "wrist_extension":
                dict["wrist_extension"] = i
            elif joint == "joint_lift":
                dict["joint_lift"] = i
        self.default_wrist_position = joint_msg.position[dict["wrist_extension"]]
        self.default_lift_position = joint_msg.position[dict["joint_lift"]]

    def calibrate_extensions_cb(self, msg):
        """
        Callback function to construct discrete set of wrist extensions and lift positions
        Perform linear regression on newest data set to fit a linear function for interpolating new data points given wrist position
        or lift position

        Parameters:
                    msg (msg) Message of type WristExtension 

        """
        if msg.lift_time == 0:
            # fit linear function y = mx + b where y is time (sec) required to extension to position (x)
            self.wrist_extension_values.append(msg.wrist_extension - self.default_wrist_position)
            self.wrist_extension_times.append(msg.push_time)

            wrist_slope, wrist_intercept, wrist_r_value, wrist_p_value, wrist_std_err = scipy.stats.linregress(self.wrist_extension_values, self.wrist_extension_times)

            self.wrist_calibration['slope'] = wrist_slope
            self.wrist_calibration['intercept'] = wrist_intercept
        else:
            # fit linear function y = mx + b where y is time (sec) required to extension to position (x)
            self.lift_position_values.append(msg.lift_position)
            self.lift_position_times.append(msg.lift_time)

            lift_slope, lift_intercept, lift_r_value, lift_p_value, lift_std_err = scipy.stats.linregress(self.lift_position_values, self.lift_position_times)
        
            self.lift_calibration['slope'] = lift_slope
            self.lift_calibration['intercept'] = lift_intercept

    def calibrate_extensions(self):
        """
        Run set number of service calls to obtain discrete set of wrist and lift extensions for linear fitting
        Throw exception when any service call fails

        """
        try:
            # send service calls to extend telescopic arm 0.03 m until some max limit
            wrist_extension = self.default_wrist_position
            max_wrist_extension = 0.51
            
            while wrist_extension + 0.03 < max_wrist_extension:
                wrist_extension += 0.03
                self.push_object_sp(wrist_extension=wrist_extension)

            # send service calls to raise telescopic arm 0.01 m until some max limit
            lift_raise = 0
            max_lift_raise = 0.1 
            while lift_raise + 0.01 < max_lift_raise:
                lift_raise += 0.01
                self.pull_object_sp(wrist_extension=self.default_wrist_position, lift_position=lift_raise)

            # publish calibration parameters (CalibrationParams.msg) for wrist extensions
            wrist_calibration_param.slope = self.wrist_calibration['slope']
            wrist_calibration_param.intercept = self.wrist_calibration['intercept']
            self.wrist_calibration_pub.publish(wrist_calibration_param)

            # publish calibration parameters (CalibrationParams.msg) for lift extensions
            lift_calibration_param.slope = self.lift_calibration['slope']
            lift_calibration_param.intercept = self.lift_calibration['intercept']
            self.lift_calibration_pub.publish(lift_calibration_param)

        except rospy.ServiceException:
            print("Service call failed")

    def aruco_marker_position_cb(self, msg):
        '''
        Callback function to track the position of the aruco marker on the belt and append position to a buffer

        Parameters:
                msg (message): 3D position of the moving aruco marker on the belt and the ID of the marker
        '''
        if msg.label == VELOCITY_MARKER_ID:
            self.aruco_marker_position_buffer.append(msg)
            self.calculate_velocity()
        self.aruco_velocity_pub.publish(self.velocity)

    def calculate_velocity(self):
        '''
        Function to calculate the current belt velocity given the current aruco marker position buffer
        '''
        msg_length = len(self.aruco_marker_position_buffer)
        first_msg = self.aruco_marker_position_buffer[0]
        last_msg = self.aruco_marker_position_buffer[msg_length-1]

        # Difference in time between when the marker was first seen and when the marker moves out of the frame
        t = last_msg.header.stamp.secs - first_msg.header.stamp.secs
        
        if msg_length > 2 and t > 0:
            # velocity = difference in the positions of the marker / time taken to cover the distance
            self.velocity = (last_msg.pose.position.x - first_msg.pose.position.x) / t

if __name__ == '__main__':
    try:
        rospy.init_node("calibration_node")
    
        rate = rospy.Rate(SLEEP_RATE)
        
        calibration_node = VelocityCalculator()
        calibration_node.calibrate_extensions()
        
        while not rospy.is_shutdown():
            calibration_node.lift_calibration_pub.publish(lift_calibration_param)
            calibration_node.wrist_calibration_pub.publish(wrist_calibration_param)
            calibration_node.aruco_velocity_pub.publish(calibration_node.velocity)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass