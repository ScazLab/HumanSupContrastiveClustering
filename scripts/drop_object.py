#!/usr/bin/env python

from __future__ import print_function

from sensor_msgs.msg import JointState

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal, FollowJointTrajectoryResult
from trajectory_msgs.msg import JointTrajectoryPoint
from recycling_stretch.msg import WristExtension
from recycling_stretch.srv import PushObject, PushObjectRequest, PushObjectResponse, \
                                PullObject, PullObjectRequest, PullObjectResponse, \
                                LiftArm, LiftArmRequest, LiftArmResponse
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

import math
import time
import threading
import sys
import time

import argparse as ap
import hello_helpers.hello_misc as hm

class DropObjectNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()

        # Common parameters
        self.initial_wrist_extension = 0.33
        self.wrist_effort = 120
        self.retraction_velocity = 0.5

        # Push parameters
        self.wrist_push_effort = 120
        self.push_wrist_extension = 0.29 # max = 0.51
        self.push_extension_velocity = 0.2

        # Pull parameters
        self.wrist_pull_effort = 120
        self.lift_raise = 0.1 # TODO: Make this dynamic based on the contents of the belt
        self.lift_effort = 20
        self.lift_velocity = 0.5
        self.pull_extension_velocity = 0.2
        self.pull_wrist_extension = 0.29

        # Publishers
        self.wrist_extension_time_pub = rospy.Publisher('/recycling_stretch/wrist_extension_time', WristExtension, queue_size=1)
        self.lift_extension_time_pub = rospy.Publisher('/recycling_stretch/lift_extension_time', WristExtension, queue_size=1)
    ######################################################################
    ################# ROBOT CONTROLLER ###################################
    ######################################################################

    def move_to_pose_acc(self, pose, velocity, acceleration, async=False):
        '''
        Move joints to a given position when position, velocity and accelerations are specified using ActionLib
        
        Parameters:
                    pose (dict): Dictionary of goal joint positions in the format {'joint_name': joint_position}
                    velocity (dict): Dictionary of goal joint velocities in the format {'joint_name': joint_velocity}
                    acceleration (dict): Dictionary of goal joint accelerations in the format {'joint_name': joint_acceleration}
                    async (bool): Boolean value indicating whether or not to wait for result from the ActionLib server
        Returns:
                    success (bool): Boolean value indicating if the goal trajectory was successfully executed
        '''
        rospy.loginfo("Calling Move to Pose with Acceleration")
        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory',
                                                              FollowJointTrajectoryAction)
        joint_names = [key for key in pose]

        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0.0)
        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = joint_names

        joint_positions = [pose[key] for key in joint_names]
        joint_velocities = [velocity[key] for key in joint_names]
        joint_accelerations = [acceleration[key] for key in joint_names]

        point.positions = joint_positions
        point.velocities = joint_velocities
        point.accelerations = joint_accelerations
        trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)

        if not async:
            self.trajectory_client.wait_for_result()

    def move_to_pose_effort(self, pose, velocity, async=False, custom_contact_thresholds=False):
        '''
        Move joints to a given position when position, velocity and efforts are specified using ActionLib
        
        Parameters:
                    pose (dict): Dictionary of goal joint positions in the format 
                                - {'joint_name': joint_position} when use_custom_contact_thresholds=False
                                - {'joint_name': (joint_position, joint_effort when use_custom_thresholds=True)}
                    velocity (dict): Dictionary of goal joint velocities in the format {'joint_name': joint_velocity}
                    async (bool): Boolean value indicating whether or not to wait for result from the ActionLib server
                    custom_contact_thresholds (bool): Boolean value indicating whether or not to use custom joint efforts
        Returns:
                    success (bool): Boolean value indicating if the goal trajectory was successfully executed
        '''
        rospy.loginfo("Calling Move to Pose with Effort")
        joint_names = [key for key in pose]
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0.0)

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(1.0)
        trajectory_goal.trajectory.joint_names = joint_names
        if not custom_contact_thresholds: 
            joint_positions = [pose[key] for key in joint_names]
            joint_velocities = [velocity[key] for key in joint_names]
            point.positions = joint_positions
            point.velocities = joint_velocities
            trajectory_goal.trajectory.points = [point]
        else:
            pose_correct = all([len(pose[key])==2 for key in joint_names])
            if not pose_correct:
                rospy.logerr("HelloNode.move_to_pose: Not sending trajectory due to improper pose. custom_contact_thresholds requires 2 values (pose_target, contact_threshold_effort) for each joint name, but pose = {0}".format(pose))
                return
            joint_positions = [pose[key][0] for key in joint_names]
            joint_efforts = [pose[key][1] for key in joint_names]
            joint_velocities = [velocity[key] for key in joint_names]
            point.positions = joint_positions
            point.velocities = joint_velocities
            point.effort = joint_efforts
            trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)

        if not async: 
            success = self.trajectory_client.wait_for_result()
            return success

    def joint_states_callback(self, joint_states):
        '''
        Callback function that reads the current joint states of the robot

        Parameters:
                    joint_states(message): Message with contents of the topic /stretch/joint_states
        '''
        with self.joint_states_lock:
            self.joint_states = joint_states
            wrist_position, wrist_velocity, wrist_effort = hm.get_wrist_state(joint_states)
            self.wrist_position = wrist_position
            self.wrist_velocity = wrist_velocity
            lift_position, lift_velocity, lift_effort = hm.get_lift_state(joint_states)
            self.lift_position = lift_position
    
    ######################################################################
    ################# FINE ROBOT MOVEMENTS ###############################
    ######################################################################

    def move_to_initial_configuration(self):
        '''
        Move the wrist extension joint to an initial position
        This function assumes that the lift has been teleoperated to an appropriate position

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        initial_pose = {'wrist_extension': (self.initial_wrist_extension, self.wrist_effort)}
        initial_velocity = {'wrist_extension': self.retraction_velocity}
        rospy.loginfo('Move to the initial configuration')
        success = self.move_to_pose_effort(initial_pose, initial_velocity, custom_contact_thresholds=True)
        return success

    def push(self, wrist_extension):
        '''
        Move the wrist extension joint from an initial position to the target push position

        Parameters:
                wrist_extension(float): Value of the target wrist extension to push the object off the conveyor belt

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        rospy.loginfo('Pushing Object')
        print("Current Wrist Position: ", self.wrist_position)
        self.wrist_target_m = wrist_extension 
        pose = {'wrist_extension': (self.wrist_target_m, self.wrist_push_effort)}
        wrist_velocity = {'wrist_extension': self.push_extension_velocity}
        success = self.move_to_pose_effort(pose, wrist_velocity, custom_contact_thresholds=True)
        return success
    
    def raise_arm(self):
        '''
        Move the lift joint from the initial position to a target position by an offset
        This function assumes that the lift has been teleoperated to an appropriate position

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        self.lift_target_m = self.lift_position + self.lift_raise
        pose = {'joint_lift':self.lift_target_m}
        lift_velocity = {'joint_lift': self.lift_velocity}
        success = self.move_to_pose_effort(pose, lift_velocity)
        return success
    
    def pull(self, wrist_extension):
        '''
        Move the wrist extension joint from an initial position to the target push position

        Parameters:
                wrist_extension(float): Value of the target wrist extension to pull the object off the conveyor belt

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        rospy.loginfo('Pulling Object')
        pose = {'wrist_extension': (wrist_extension, self.wrist_pull_effort)}
        wrist_velocity = {'wrist_extension': self.pull_extension_velocity}
        success = self.move_to_pose_effort(pose, wrist_velocity, custom_contact_thresholds=True)
        return success

    def lower_arm(self):
        '''
        Move the lift joint from the target position back to an initial position
        This function assumes that the lift has been teleoperated to an appropriate position

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        # TODO: Add a tolerance value. The arm does not come back exactly to the same position
        self.lift_target_m = self.lift_position - self.lift_raise 
        pose = {'joint_lift':self.lift_target_m}
        lift_velocity = {'joint_lift': self.lift_velocity}
        success = self.move_to_pose_effort(pose, lift_velocity)
        return success
    
    ######################################################################
    ################# COMPOSITE ROBOT MOVEMENTS ##########################
    ######################################################################

    def push_object(self, wrist_extension):
        '''
        Set of movements to push an object from the conveyor belt called in the following order:
            1. Extend arm from initial position to target position
            2. Retract arm back to initial position

        Parameters:
                wrist_extension(float): Value of the target wrist extension to push the object off the conveyor belt

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        rospy.loginfo('Pushing Object')
        print("Current Wrist Position: ", self.wrist_position)
        start_time = time.time()

        ###### Extend arm #######
        print("Before Push: ", self.wrist_position)
        push_success = self.push(wrist_extension)
        push_time = time.time() - start_time
        print("Push time:", push_time)
        print("Final position: ", self.wrist_position)

        ###### Retract arm #######
        retract_success = self.move_to_initial_configuration()
        rospy.loginfo('Arm Retracted')
        end_time = time.time() - start_time
        print("Time taken to retract:", end_time)

        ###### Publish wrist extension values #######
        wrist_extension_msg = WristExtension()
        wrist_extension_msg.wrist_extension = wrist_extension
        wrist_extension_msg.push_time = push_time
        wrist_extension_msg.retraction_time = end_time
        self.wrist_extension_time_pub.publish(wrist_extension_msg)

        # Sanity Check
        if push_success and retract_success:
            success = True
        else:
            success = False
        return success
    
    def pull_object(self, wrist_extension):
        '''
        Set of movements to pull an object from the conveyor belt called in the following order:
            1. Lift arm such that it is above all objects
            2. Extend arm from initial position to target position
            3. Retract arm back to initial position
            4. Drop arm down to original lift position

        Parameters:
                wrist_extension(float): Value of the target wrist extension to push the object off the conveyor belt

        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        rospy.loginfo('Pulling Object')
        start_time = time.time()
        print("Initial lift position: ", self.lift_position)

        ###### Move arm up ########
        raise_success = self.raise_arm()
        lift_time = time.time() - start_time
        print("Raised position: ", self.lift_position)

        ###### Extend arm ##########
        pull_success = self.pull(wrist_extension)
        print("Extended position: ", self.wrist_position)
        pull_time = time.time() - start_time
        print("Pull time: ", pull_time)

        ####### Retract arm ##########
        retract_success = self.move_to_initial_configuration()
        print("Retracted position: ", self.wrist_position)

        ####### Drop arm down #######
        lower_success = self.lower_arm()
        print("Lowered position: ", self.lift_position)
        end_time = time.time() - start_time
        print("Pull time: ", end_time)

        ###### Publish wrist extension values #######
        wrist_extension_msg = WristExtension()
        wrist_extension_msg.wrist_extension = wrist_extension
        wrist_extension_msg.push_time = pull_time
        wrist_extension_msg.retraction_time = end_time
        wrist_extension_msg.lift_time = lift_time
        wrist_extension_msg.lift_position = self.lift_raise
        self.wrist_extension_time_pub.publish(wrist_extension_msg)

        if raise_success and pull_success and lower_success and retract_success:
            success = True
        else:
            success = False
        return success

    def lift_arm(self):
        ###### Move arm up ########
        raise_success = self.raise_arm()
        print("Raised position: ", self.lift_position)
        
    ######################################################################
    ################# SERVICE CALLBACKS ##################################
    ######################################################################

    def push_object_srv_cb(self, srv):
        '''
        Service callback to trigger push_object function

        Parameters:
                srv (service): Contents of the /recycling_stretch/push_object_srv service
        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        self.push_wrist_extension = srv.wrist_extension
        success = self.push_object(self.push_wrist_extension)
        return PushObjectResponse(success)

    def pull_object_srv_cb(self, srv):
        '''
        Service callback to trigger pull_object function

        Parameters:
                srv (service): Contents of the /recycling_stretch/pull_object_srv service
        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        self.pull_wrist_extension = srv.wrist_extension
        self.lift_raise = srv.lift_extension
        success = self.pull_object(self.pull_wrist_extension)
        return PullObjectResponse(success)
    
    def lift_arm_srv_cb(self, srv):
        '''
        Service callback to trigger lift_arm function

        Parameters:
                srv (service): Contents of the /recycling_stretch/lift_arm_srv service
        Returns:
                success(bool): Boolean value indicating whether or not the movement was compeleted successfully
        '''
        self.lift_raise = srv.lift_extension
        success = self.lift_arm()
        return LiftArmResponse(success)
    ########################################################################

    def main(self):
        hm.HelloNode.main(self, 'drop_object', 'drop_object', wait_for_first_pointcloud=False)
        self.initial_joint_state_calibration = rospy.wait_for_message('/stretch/joint_states', JointState)
        initial_wrist_extension, wrist_velocity, wrist_effort = hm.get_wrist_state(self.initial_joint_state_calibration)
        self.initial_wrist_extension = initial_wrist_extension
        print("initial_wrist_extension {}".format(initial_wrist_extension))

        self.joint_states_sub = rospy.Subscriber('/stretch/joint_states', JointState, self.joint_states_callback)

        self.push_object_srv = rospy.Service('/recycling_stretch/push_object_srv', PushObject, self.push_object_srv_cb)
        self.pull_object_srv = rospy.Service('/recycling_stretch/pull_object_srv', PullObject, self.pull_object_srv_cb)
        self.lift_arm_srv = rospy.Service('/recycling_stretch/lift_arm_srv', LiftArm, self.lift_arm_srv_cb)

        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            rate.sleep()


if __name__ == '__main__':
    try:
        node = DropObjectNode()
        node.main()
    except KeyboardInterrupt:
        rospy.loginfo('interrupt received, so shutting down')
#    except rospy.ROSInterruptException:
#        pass
# rospy.loginfo('keyboard_teleop was interrupted', file=sys.stderr)