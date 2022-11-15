#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from recycling_stretch.msg import DetectedBox, RotatedBBox, DebugDetectedBox
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Int64MultiArray, Float64MultiArray, MultiArrayDimension
import ros_numpy
import matplotlib.pyplot as plt
from PIL import Image as ImageShow
import sys
sys.path.append('/home/scazlab/catkin_ws/src/mmdetection/scripts')
from mmdet_inference import run_inference
from mmdet.apis import init_detector
import os
import math

# Root Paths
CONFIG_PARENT_PATH = '/home/scazlab/catkin_ws/src/mmdetection/configs/recycling'
MODEL_PARENT_PATH = '/home/scazlab/catkin_ws/src/mmdetection/work_dirs/yolact_r101_1x8_recycling'

# Visualization Parameters
VIZ = True
BBOX_THICKNESS = 2
COLOR = (0, 255, 0)
DRAW_AA_BBOX = False
DRAW_RBBOX = True


class RealTimeObjectDetection():
    """ROS node that detects recyclables in real time"""

    def __init__(self):

        # Init the node
        rospy.init_node('real_time_object_detection')

        # Node parameters
        self.model_name = 'epoch_100.pth'
        self.config_file = 'yolact_r101_1x8_recycling.py'
        self.model_path = os.path.join(MODEL_PARENT_PATH, self.model_name)
        self.config_path = os.path.join(CONFIG_PARENT_PATH, self.config_file)
        self.num_classes = 4
        self.num_frame_skips = 1 # assuming 30fps

        self.class_map = {1: 'Can', 2: 'Bottle', 3: 'Milk Jug', 4: 'Cardboard'}
        self.score_threshold_dict =  {1: 0.5126, 2: 0.8644, 3: 0.1859, 4: 0.1859} 

        # build the model from a config file and a checkpoint file
        self.model = init_detector(self.config_path, self.model_path, device='cuda')

        # Publishers
        self.detection_image_pub = rospy.Publisher('/recycling_stretch/detection_image', Image, queue_size=5)
        self.detection_box_pub = rospy.Publisher('/recycling_stretch/detection_boxes', DebugDetectedBox, queue_size=10)
       
        # Subscribers
        # self.camera_info_sub = message_filters.Subscriber('/camera2/color/camera_info', CameraInfo)
        # self.rgb_img_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)
        # self.depth_img_sub = message_filters.Subscriber('/camera2/aligned_depth_to_color/image_raw', Image)

        # self.synchronizer = message_filters.TimeSynchronizer(
        #     [self.rgb_img_sub, self.depth_img_sub], 10)
        # self.synchronizer.registerCallback(self.get_image_cb)

        self.rgb_image_sub = rospy.Subscriber('/camera2/color/image_raw', Image, self.get_image_cb)

        self.count = 0
        self.img_height = 720
        self.img_width = 1280

        self.detected_box_msg = DebugDetectedBox()

        # main thread just waits now..
        rospy.spin()

    def camera_info_cb(self, msg):
        print("getting camera info")
        print(msg)

    def get_image_cb(self, ros_rgb_image):
        self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp

        # print("TIME OF MESSAGE (ROSTIME) {}".format(ros_rgb_image.header.stamp))

        # self.depth_image = ros_numpy.numpify(ros_depth_image)
        # self.depth_image_timestamp = ros_depth_image.header.stamp

        # OpenCV expects bgr images, but numpify by default returns rgb images.
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)


        # time_diff = self.rgb_image_timestamp - self.depth_image_timestamp
        # time_diff = abs(time_diff.to_sec())
        # if time_diff > 0.0001:
        #     print('WARNING: The rgb image and the depth image were not taken at the same time.')
        #     print('         The time difference between their timestamps =', time_diff, 's')
        
        self.count = self.count + 1

        if self.count % self.num_frame_skips == 0:
            detection_bboxes, detection_classes, detection_scores, detection_rboxes = self.detect_object()
            if len(detection_bboxes) != len(detection_rboxes):
                print("Misalignment")
                return
            else:
                if VIZ:
                    detection_image = self.visualize_bboxes(detection_bboxes, detection_classes, detection_scores, detection_rboxes)

                    if detection_image is not None:
                        try:
                            detection_image_msg = self.cv2_to_imgmsg(detection_image)
                        except CvBridgeError as e:
                            rospy.logerr(e)
                            return

                        detection_image_msg.header = ros_rgb_image.header
                        self.detection_image_pub.publish(detection_image_msg)
                        self.publish_box_info(detection_bboxes, detection_classes, detection_scores, detection_rboxes, detection_image_msg)
                    else:
                        print("No Object!!!")
    
    def detect_object(self):
        '''
        Wrapper over the object detector. Runs inference on the trained model

        Parameters:
                    None
        Returns:
                    all_boxes (list(list)): List of all axis aligned bounding boxes in the [(x_min, y_min), (x_max, y_max)] format
                    all_classes (list): List of all the predicted classes
                    all_scores (list): List of all confidence scores for all the predicted boxes
                    all_rboxes (list(list)): List of coordinates for rotated bounding boxes derived from predicted masks
                                             in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)] format
        '''
        all_bboxes, all_classes, all_scores, all_rboxes = run_inference(self.model, self.rgb_image, self.num_classes)
        return all_bboxes, all_classes, all_scores, all_rboxes
    
    def publish_box_info(self, detection_bboxes, detection_classes, detection_scores, detection_rboxes, detection_image):
        '''
        Iterates through all the boxes of each category, fiters out boxes lower than the best thresholds 
        and publish that to the /recycling_stretch/detection_boxes topic. Optionally, visualize the predicted boxes on the image
        on the /recycling_stretch/detection_image topic

        Parameters:
                    detection_boxes (list(list)): List of all axis aligned bounding boxes in the [(x_min, y_min), (x_max, y_max)] format
                    detection_classes (list): List of all the predicted classes
                    detection_scores (list): List of all confidence scores for all the predicted boxes
                    detection_rboxes (list(list)): List of coordinates for rotated bounding boxes derived from predicted masks
                                                   in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)] format
        Returns:
                None
        '''
        bb_roi_msg_list = []
        rbb_roi_msg_list = []
        scores_list = []
        class_list = []

        for i, _ in enumerate(detection_scores):
            score = detection_scores[i]
            class_id = detection_classes[i]
            threshold = self.score_threshold_dict[class_id]
            score = np.round(score, 2)

            # Filter boxes with confidence lower than the threshold for each class
            if score < threshold:
                continue

            '''
            Calculate box parameters for axis aligned boxes
            '''
            x_top = detection_bboxes[i][0]
            y_top = detection_bboxes[i][1]
            height = detection_bboxes[i][2] - detection_bboxes[i][0]
            width = detection_bboxes[i][3] - detection_bboxes[i][1]

            bb_roi_msg = self.form_roi_msg(x_top, y_top, height, width)
            bb_roi_msg_list.append(bb_roi_msg)

            '''
            Calculate box parameters for rotated boxes
            '''
            lt_x = abs(detection_rboxes[i][0][0])
            lt_y = abs(detection_rboxes[i][0][1])
            rt_x = abs(detection_rboxes[i][1][0])
            rt_y = abs(detection_rboxes[i][1][1])
            lb_x = abs(detection_rboxes[i][2][0])
            lb_y = abs(detection_rboxes[i][2][1])
            rb_x = abs(detection_rboxes[i][3][0])
            rb_y = abs(detection_rboxes[i][3][1])

            rbb_roi_msg = self.form_rbbox_msg(lt_x, lt_y, rt_x, rt_y, lb_x, lb_y, rb_x, rb_y)
            rbb_roi_msg_list.append(rbb_roi_msg)

            scores_list.append(score)
            class_list.append(class_id)
        
        self.detected_box_msg.header.stamp = self.rgb_image_timestamp
        self.detected_box_msg.img = detection_image
        self.detected_box_msg.bboxes =  bb_roi_msg_list
        self.detected_box_msg.rbboxes =  rbb_roi_msg_list
        self.detected_box_msg.pred_category = class_list
        self.detected_box_msg.scores = scores_list
        self.detection_box_pub.publish(self.detected_box_msg)
    
    def form_rbbox_msg(self, lt_x, lt_y, rt_x, rt_y, lb_x, lb_y, rb_x, rb_y):
        '''
        Format the bounding box predictions into the custom RotatedBBox format message

        Parameters:
                    lt_x (int): top-left x-coordinate of the box
                    lt_y (int): top-left y-coordinate of the box
                    rt_x (int): top-right x-coordinate of the box
                    rt_y (int): top-right y-coordinate of the box
                    lb_x (int): bottom-left x-coordinate of the box
                    lb_y (int): bottom-left y-coordinate of the box
                    rb_x (int): bottom-right x-coordinate of the box
                    rb_y (int): bottom-right y-coordinate of the box
                    
        Returns:
                    rbbox_msg (message): rotated bounding box message
        '''

        rbbox_msg = RotatedBBox()
        rbbox_msg.top_left_x = int(lt_x)
        rbbox_msg.top_left_y = int(lt_y)
        rbbox_msg.top_right_x = int(rt_x)
        rbbox_msg.top_right_y = int(rt_y)
        rbbox_msg.bottom_left_x = int(lb_x)
        rbbox_msg.bottom_left_y = int(lb_y)
        rbbox_msg.bottom_right_x = int(rb_x)
        rbbox_msg.bottom_right_y = int(rb_y)
        return rbbox_msg

    def form_roi_msg(self, x_top, y_top, height, width):
        '''
        Format the bounding box predictions into the RegionOfInterest message
        https://answers.ros.org/question/270649/how-can-i-publish-many-rois-in-a-detection-node/

        Parameters:
                    x_top (int): x coordinate of the top left corner
                    y_top (int): y coordinate of the top left corner
                    height (int): height of the bounding box
                    width (int): width of the bounding box
        Returns:
                    bb_roi_msg (message): bounding box message
        '''
        bb_roi_msg = RegionOfInterest()
        bb_roi_msg.x_offset = x_top
        bb_roi_msg.y_offset = y_top
        bb_roi_msg.height = height
        bb_roi_msg.width = width
        return bb_roi_msg
    
    def visualize_bboxes(self, detection_bboxes, detection_classes, detection_scores, detection_rboxes):
        '''
        Vizualize axis-aligned and rotated bounding boxes on the image

        Parameters:
                    detection_boxes (list(list)): List of all axis aligned bounding boxes in the [(x_min, y_min), (x_max, y_max)] format
                    detection_classes (list): List of all the predicted classes
                    detection_scores (list): List of all confidence scores for all the predicted boxes
                    detection_rboxes (list(list)): List of coordinates for rotated bounding boxes derived from predicted masks
                                                   in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)] format
        Returns:
                    detection_image (image): Image with bounding boxes drawn on the image
        '''
        detection_image = self.rgb_image 
        for i, _ in enumerate(detection_scores):
            score = detection_scores[i]
            class_id = detection_classes[i]
            threshold = self.score_threshold_dict[class_id]
            score = np.round(score, 2)

            if score < threshold:
                continue

            if DRAW_AA_BBOX:
                detection_image = self.draw_aa_boxes(self.rgb_image, detection_bboxes[i], class_id, COLOR, BBOX_THICKNESS, put_text=True, put_score=True)

            if DRAW_RBBOX:
                detection_image = self.draw_rboxes(self.rgb_image, detection_rboxes[i], class_id, COLOR, BBOX_THICKNESS, score, put_score= False)
        return detection_image
    
    def draw_aa_boxes(self, image, box, class_id, color, thickness, put_text, score, put_score=True):
        '''
        Helper function to draw axis aligned bounding boxes on the image

        Parameters:
                    image (image): Input image from get_image_cb
                    box (list): Coordinates of a bounding box in the [x_min, y_min, x_max, y_max] format
                    class_id (int): ID of the predicted class
                    color (tuple): Color of the bounding box
                    thickness (int): Thickness of the bounding box
                    score (float):  Confidence score of the bounding box
                    put_score (bool): Boolean flag to decide whether to print the score on the bounding box
        Returns:
                    image (image): Image with the bounding boxes plotted on the image
        '''
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])

        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        if put_text:
            cv2.putText(image,self.class_map[class_id],(start_point),cv2.FONT_HERSHEY_SIMPLEX,1, color, thickness)
        if put_score:
            cv2.putText(image,str(score),(start_point),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        return image

    def draw_rboxes(self, image, rboxes, class_id, color, thickness, put_text, score=None, put_score=True):
        '''
        Helper function to draw rotated bounding boxes on the image

        Parameters:
                    image (image): Input image from get_image_cb
                    rboxes (list(tuples)): Coordinates of a bounding box in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format
                    class_id (int): ID of the predicted class
                    color (tuple): Color of the bounding box
                    thickness (int): Thickness of the bounding box
                    score (float):  Confidence score of the bounding box
                    put_score (bool): Boolean flag to decide whether to print the score on the bounding box
        Returns:
                    image (image): Image with the bounding boxes plotted on the image
        '''
        cv2.line(image, (rboxes[0][0], rboxes[0][1]), (rboxes[1][0], rboxes[1][1]), color, thickness)
        cv2.line(image, (rboxes[1][0], rboxes[1][1]), (rboxes[2][0], rboxes[2][1]), color, thickness)
        cv2.line(image, (rboxes[2][0], rboxes[2][1]), (rboxes[3][0], rboxes[3][1]), color, thickness)
        cv2.line(image, (rboxes[3][0], rboxes[3][1]), (rboxes[0][0], rboxes[0][1]), color, thickness)

        put_text=False
        if put_text:
            cv2.putText(image, self.class_map[class_id],(rboxes[0][0],rboxes[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        if put_score:
            cv2.putText(image,str(score),(rboxes[1][0],rboxes[1][1]),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        return image

    def cv2_to_imgmsg(self, cv_image):
        '''
        Helper function to publish a cv2 image as a ROS message (without using ROS cv2 Bridge)
        https://answers.ros.org/question/350904/cv_bridge-throws-boost-import-error-in-python-3-and-ros-melodic/

        Parameters:
                    cv_image (image): Image to publish to a ROS message
        Returns:
                    img_msg (message): Image message published to a topic 
        '''
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
        return img_msg


if __name__ == '__main__':
    try:
        node = RealTimeObjectDetection()
    except rospy.ROSInterruptException:
        pass
