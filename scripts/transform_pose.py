#!/usr/bin/env python2

import rospy  # If you are doing ROS in python, then you will always need this import
import os
# Message imports go here
import geometry_msgs.msg
from sensor_msgs.msg import Image
from recycling_stretch.msg import RealPose, ArucoRefMarker
from geometry_msgs.msg import PoseStamped
# Service imports go here

# All other imports go here
import tf
import tf2_ros
import tf2_geometry_msgs


# Hyper-parameters go here
SLEEP_RATE = 10


class TransformPose():
    """
    Note: In the init function, I will first give a description of each part, and then I will give an example
    """

    def __init__(self):
        # Everything besides pubs, subs, services, and service proxies go here
        print("Starting!!!")

        # Publishers go here
        self.transformed_real_pose_pub = rospy.Publisher('/recycling_stretch/transformed_real_pose', RealPose, queue_size=10)
        self.transformed_marker_pose_pub = rospy.Publisher('/recycling_stretch/transformed_marker_pose', ArucoRefMarker, queue_size=10)

        # Service Proxies go here

        # Subscribers go here
        self.image_sub = rospy.Subscriber('/camera2/color/image_raw', Image, self.get_image_cb)
        #self.real_pose_sub = rospy.Subscriber('/recycling_stretch/real_pose', RealPose, self.real_pose_cb)
        #self.marker_pose_sub = rospy.Subscriber('/recycling_stretch/marker_position', ArucoRefMarker, self.aruco_marker_cb)
        self.object_pose_time = rospy.Time.now()
        # Services go here

        self.camera_offset = 0.635
        # self.camera_offset_1_2 = 0.631825
        # self.camera_offset_2_3 = 0.6731
        # self.camera_offset = self.camera_offset_1_2 + self.camera_offset_2_3
        rospy.spin()

    def get_image_cb(self, msg):
        
        self.image_time = msg.header.stamp
        self.static_transform_broadcaster("camera1_link", "camera2_link", 0, self.camera_offset, 0, 0, 0, 0)
        self.static_transform_broadcaster("base_link", "camera1_link", -1, -1, 1, 0, 1.57, 0)

    '''
    def real_pose_cb(self, msg):


        trp = RealPose()
        trp.sim_detected_box_data = msg.sim_detected_box_data
        poses = msg.poses
        for pose in poses:
            self.static_transform_broadcaster("camera1_link", "camera2_link", 0, self.camera_offset, 0, 0, 0, 0)
            self.static_transform_broadcaster("base_link", "camera1_link", -1, -1, 1, 0, 1.57, 0)
            transformed_pose = self.transform_pose(pose.pose, "camera2_link", "camera1_link")
            ps = PoseStamped()
            self.object_pose_time = rospy.Time(0) # msg.header.stamp
            ps.header.stamp = rospy.Time.now()
            ps.header.frame_id = "camera1_link"
            ps.pose.position.x = transformed_pose.position.x
            ps.pose.position.y = transformed_pose.position.y
            ps.pose.position.z = transformed_pose.position.z

            trp.poses.append(ps)
        self.transformed_real_pose_pub.publish(trp)
    
    def aruco_marker_cb(self, msg):

        self.static_transform_broadcaster("camera1_link", "camera2_link", 0, self.camera_offset, 0, 0, 0, 0)
        self.static_transform_broadcaster("base_link", "camera1_link", -1, -1, 1, 0, 1.57, 0)
        tmp = ArucoRefMarker()
        pose = msg.pose
        transformed_pose = self.transform_pose(pose, "camera1_color_optical_frame", "camera1_link")
        tmp.header = msg.header
        tmp.label = msg.label
        tmp.pose = transformed_pose
        tmp.detected = msg.detected
        self.transformed_marker_pose_pub.publish(tmp)
    '''

    def static_transform_broadcaster(self, parent_frame, child_frame, x, y, z, roll, pitch, yaw):
        broadcaster = tf2_ros.StaticTransformBroadcaster()

        static_transformStamped = geometry_msgs.msg.TransformStamped()

        static_transformStamped.header.stamp = self.image_time # rospy.Time(0)
        static_transformStamped.header.frame_id = parent_frame
        static_transformStamped.child_frame_id = child_frame

        static_transformStamped.transform.translation.x = x
        static_transformStamped.transform.translation.y = y
        static_transformStamped.transform.translation.z = z

        quat = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        static_transformStamped.transform.rotation.x = quat[0]
        static_transformStamped.transform.rotation.y = quat[1]
        static_transformStamped.transform.rotation.z = quat[2]
        static_transformStamped.transform.rotation.w = quat[3]
        # print(static_transformStamped)
        broadcaster.sendTransform(static_transformStamped)
    
    def transform_pose(self, input_pose, from_frame, to_frame):
        """
        https://answers.ros.org/question/323075/transform-the-coordinate-frame-of-a-pose-from-one-fixed-frame-to-another/
        """
        # **Assuming /tf2 topic is being broadcasted
        tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(720.0))
        listener = tf2_ros.TransformListener(tf_buffer)

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = from_frame
        pose_stamped.header.stamp = rospy.Time(0)# rospy.Time.now()

        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = tf_buffer.transform(pose_stamped, to_frame, rospy.Duration(5))
            return output_pose_stamped.pose

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            os._exit(1)
            raise
            


def main():
    rospy.init_node("transform_pose")

    transform_pose = TransformPose()

    rate = rospy.Rate(SLEEP_RATE)

    # This while loop is what keeps the node from dieing
    while not rospy.is_shutdown():
        # If I wanted my node to constantly publish something, I would put the publishing here

        rate.sleep()


if __name__ == '__main__':
    main()