#!/usr/bin/env python

import cv2
import pdb
import numpy as np
import rospy
from recycling_stretch.msg import DetectedBox, RealPose, DebugSimDetectedBox, DebugRealPose
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import CameraInfo
import ros_numpy
import matplotlib.pyplot as plt

# from PIL import Image
from geometry_msgs.msg import PoseStamped, Pose, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import math
import sys

# global variables
BELT_DEPTH = 800
class ConvertBBox2Real():
    def __init__(self):

        self.last_avg_depth = None
        self.last_x = None
        self.last_y = None

        # Init the node
        rospy.init_node('convert_bbox2real')

        # Publisher
        self.pose_pub = rospy.Publisher("/recycling_stretch/real_pose", DebugRealPose, queue_size=5)
        self.visualize_markers_pub = rospy.Publisher('/recycling_stretch/object_marker_array', MarkerArray, queue_size=1)

        # Subscriber
        self.camera_info_sub = message_filters.Subscriber('/camera2/color/camera_info', CameraInfo)
        self.rgb_img_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)
        self.depth_img_sub = message_filters.Subscriber('/camera2/aligned_depth_to_color/image_raw', Image)

        # self.bbox_pos_sub = message_filters.Subscriber('/recycling_stretch/detection_boxes', DetectedBox)
        self.bbox_pos_sub = message_filters.Subscriber('/recycling_stretch/sim_detection_boxes', DebugSimDetectedBox)

        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_img_sub, self.depth_img_sub, self.camera_info_sub, self.bbox_pos_sub], 50, 100) 

        self.synchronizer.registerCallback(self.get_image_cb)

        # main thread just waits now..
        rospy.spin()

    def camera_info_cb(self, msg):
        print("getting camera info")
        print(msg)

    def get_image_cb(self, ros_rgb_image, ros_depth_image, rgb_camera_info, bbox_pos):
        print("here")

        self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp

        self.depth_image = ros_numpy.numpify(ros_depth_image)
        self.depth_image_timestamp = ros_depth_image.header.stamp

        # OpenCV expects bgr images, but numpify by default returns rgb images.
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

        # Convert rgb image message back to cv2
        cv_image = self.imgmsg_to_cv2(ros_rgb_image)

        # Convert depth image message back to cv2
        depth_image = self.depth_imgmsg_to_cv2(ros_depth_image)

        time_diff = self.rgb_image_timestamp - self.depth_image_timestamp
        time_diff = abs(time_diff.to_sec())
        if time_diff > 0.0001:
            print('WARNING: The rgb image and the depth image were not taken at the same time.')
            print('         The time difference between their timestamps =', time_diff, 's')

        # Extract camera parameters
        cx = rgb_camera_info.K[2]
        fx = rgb_camera_info.K[0]
        cy = rgb_camera_info.K[5]
        fy = rgb_camera_info.K[4]

        # Output of the object detector
        top_x_list, top_y_list, width_list, height_list = self.process_bboxes(bbox_pos)

        rp = DebugRealPose()
        rp.header.stamp = bbox_pos.header.stamp
        rp.img = bbox_pos.img
        rp.header.frame_id = "camera2_color_optical_frame"

        marker_array = MarkerArray()
        ignored_objects = []
        counter_ignore = 0
        counter_append = 0
        for i, (obj_x, obj_y, obj_w, obj_h) in enumerate(zip(top_x_list, top_y_list, width_list, height_list)):
            mask_width = obj_w
            mask_height = obj_h 
            mask_start_x = obj_x 
            mask_start_y = obj_y

            # Crop out mask from the depth image
            mask = depth_image[mask_start_y: mask_start_y + mask_height, mask_start_x: mask_start_x + mask_width]
            
            # Calculate the average depth of the object
            avg_depth = np.average(mask[mask > 0]) if np.sum(mask) > 0 else self.last_avg_depth

            # Calculate the center of the bounding box
            center_x = obj_x + obj_w//2
            center_y = obj_y + obj_h//2
            
            print("AVERAGE DEPTH IS {}".format(avg_depth))
            print("CENTER OF THE OBJECT X: {} Y: {}".format(center_x, center_y))

            if avg_depth is not None and avg_depth < BELT_DEPTH and center_x > 45:  # to deal with the case when object is not in field of view
                # Use the pin hole camera model to calculate the real world position of the objects
                big_x = ((center_x * avg_depth) - (cx * avg_depth)) / fx
                big_y = ((center_y * avg_depth) - (cy * avg_depth)) / fy
                big_z = avg_depth

                # Assign the real world position of the center of the object (in meters) to a PoseStamped message
                ps = PoseStamped()
                ps.header.stamp = bbox_pos.header.stamp
                ps.header.frame_id = "camera2_color_optical_frame"
                ps.pose.position.x = big_x / 1000
                ps.pose.position.y = big_y / 1000
                ps.pose.position.z = big_z / 1000

                rp.poses.append(ps)
                # Calculate the distance of the object from the center of the camera_color_optical_frame
                distance = self.get_distance(ps)
                print("object " + str(i) + " distance " + str(distance) + " meters")

                # Visualize the real world position of the object in RVIZ
                marker = self.get_ros_marker(ps, i, mask_width, mask_height)
                marker_array.markers.append(marker)
            else:
                ignored_objects.append(i)

        ignored_objects = sorted(ignored_objects, reverse=True)
        print(ignored_objects)
        # filter objects by depth from original detection boxes
        for item in ignored_objects:
            if item < len(bbox_pos.bboxes) or item < len(bbox_pos.rbboxes):
                try:
                    scores = list(bbox_pos.scores)
                    scores.pop(item)
                    
                    pred_category = list(bbox_pos.pred_category)
                    pred_category.pop(item)

                    similarity = list(bbox_pos.similarity)
                    similarity.pop(item)

                    bbox_pos.scores = scores
                    bbox_pos.pred_category = pred_category
                    bbox_pos.similarity = similarity

                    bbox_pos.bboxes.pop(item)
                    bbox_pos.rbboxes.pop(item)
                except:
                    raise Exception("Index does not exist in list {}".format(item))

        rp.sim_detected_box_data = bbox_pos
        if len(rp.sim_detected_box_data.rbboxes) != len(rp.poses):
            print("BBOX_LENGTH {}".format(len(rp.sim_detected_box_data.rbboxes)))
            print("Not aligned")
        self.pose_pub.publish(rp)
        self.visualize_markers_pub.publish(marker_array)
    
    def process_bboxes(self, bbox_pos):
        '''
        Read the incoming predicted bounding box messages for every frame of the incoming video
        and process them to lists

        Parameters:
                    bbox_pos (message): Message in the DetectedBox format
        Returns:
                    top_x_list (list): List of all top x coordinates of each predicted bounding box
                    top_y_list (list): List of all top y coordinates of each predicted bounding box
                    width_list (list): List of all widths of each predicted bounding box
                    height_list (list): List of all heights of each predicted bounding box
        '''
        bboxes = bbox_pos.bboxes
        top_x_list = []
        top_y_list = []
        width_list = []
        height_list = []
        for box in bboxes:
            top_x_list.append(box.x_offset)
            top_y_list.append(box.y_offset)
            width_list.append(box.width)
            height_list.append(box.height)
        return (top_x_list, top_y_list, width_list, height_list)

    @staticmethod
    def get_distance(pose):
        '''
        Helper function to calculate distance between the center of the camera's frame and the predicted position of the objects

        Parameters:
                    pose (message): Message in the PoseStamped format
        Returns:
                    distance between the pose of the object the center of the camera_color_optical_frames assumed to be at (0, 0, 0)
        '''
        return math.sqrt(pose.pose.position.x ** 2 + pose.pose.position.y ** 2 + pose.pose.position.z ** 2)

    def get_ros_marker(self, ps, object_id, width, height):
        '''
        Helper function to vizualize the real-world position of the object in RVIZ

        Parameters:
                    ps (message): PoseStamped message with the real-world position of the object
                    object_id (int): Unique ID of the object
                    width (int): Width of the object in the pixel-space
                    height (int): Height of the object in the pixel-space
        Return:
                    marker (message): Marker message representing the vizualization of the object in RVIZ
        '''
        self.marker = Marker()
        self.marker.type = self.marker.CUBE
        self.marker.action = self.marker.ADD
        self.marker.lifetime = rospy.Duration(0.2)
        self.marker.text = "Detected Object"
        self.marker.header.frame_id = "camera2_color_optical_frame"
        self.marker.header.stamp = rospy.Time.now()
        self.marker.id = object_id

        self.marker.scale.x = height/1000
        self.marker.scale.y = width/1000
        self.marker.scale.z = 0.005 # half a centimeter tall

        self.marker.color.r = 255
        self.marker.color.g = 0
        self.marker.color.b = 0
        self.marker.color.a = 0.33

        self.marker.pose.position.x = ps.pose.position.x
        self.marker.pose.position.y = ps.pose.position.y
        self.marker.pose.position.z = ps.pose.position.z

        self.marker.pose.orientation.x = ps.pose.orientation.x
        self.marker.pose.orientation.y = ps.pose.orientation.y
        self.marker.pose.orientation.z = ps.pose.orientation.z
        self.marker.pose.orientation.w = ps.pose.orientation.w        

        return self.marker
    
    def imgmsg_to_cv2(self, img_msg):
        '''
        Helper function to convert a ROS RGB Image message to a cv2 image (without using ROS cv2 Bridge)
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/

        Parameters:
                    img_msg (message): Image message published to a topic 
        Returns:
                    cv_image (image): cv2 image
        '''
        dtype = np.dtype("uint8") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

    def depth_imgmsg_to_cv2(self, img_msg):
        '''
        Helper function to convert a ROS depth Image message to a cv2 image (without using ROS cv2 Bridge)
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/

        Parameters:
                    img_msg (message): Image message published to a topic 
        Returns:
                    cv_image (image): cv2 image
        '''
        dtype = np.dtype("int16") # Hardcode to 8 bits...
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 1), # and one channel of data. 
                        dtype=dtype, buffer=img_msg.data)
        # If the byt order is different between the message and the system.
        if img_msg.is_bigendian == (sys.byteorder == 'little'):
            image_opencv = image_opencv.byteswap().newbyteorder()
        return image_opencv

if __name__ == '__main__':
    try:
        node = ConvertBBox2Real()
    except rospy.ROSInterruptException:
        pass
