#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from recycling_stretch.msg import DetectedBox, PositiveCropProperties, NegativeCropProperties, NegativeCrops
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import message_filters
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Int64MultiArray, Float64MultiArray, MultiArrayDimension
import ros_numpy
import matplotlib.pyplot as plt
from PIL import Image as ImageShow
import os
import math
import torch
import torch.nn as nn
from shapely.geometry import Polygon, Point

# mmdetection object detector
import sys
sys.path.append('/home/scazlab/catkin_ws/src/mmdetection/scripts')
from mmdet_inference import run_inference
from mmdet.apis import init_detector

# hand keypoint detector
from hand_tracker.hand_detector import HandDetector

# hand pose classifier
from hand_tracker.classifier.train import ImageClassifier

import pdb

# Object Detector Model Paths
CONFIG_PARENT_PATH = '/home/scazlab/catkin_ws/src/mmdetection/configs/recycling'
OBJECT_DETECT_MODEL_PARENT_PATH = '/home/scazlab/catkin_ws/src/mmdetection/work_dirs/yolact_r101_1x8_recycling'

# Hand Pose Classifier Model Path
HAND_POSE_MODEL_PARENT_PATH = '/home/scazlab/catkin_ws/src/recycling_stretch/scripts/hand_tracker/classifier/weights/v2/'

# # Visualization Parameters
VIZ = True
BBOX_THICKNESS = 2
COLOR = (0, 255, 0)
DRAW_AA_BBOX = False
DRAW_RBBOX = True

class FindPickedObject():
    def __init__(self):

        # Init the node
        rospy.init_node('find_picked_object_node')

        # Node parameters
        self.object_detect_model_name = 'epoch_100.pth'
        self.config_file = 'yolact_r101_1x8_recycling.py'
        self.hand_pose_model_name = 'epoch_20.pth'
        self.object_detect_model_path = os.path.join(OBJECT_DETECT_MODEL_PARENT_PATH, self.object_detect_model_name)
        self.config_path = os.path.join(CONFIG_PARENT_PATH, self.config_file)
        self.hand_pose_model_path = os.path.join(HAND_POSE_MODEL_PARENT_PATH, self.hand_pose_model_name)
        self.device = 'cuda'

        self.mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        self.num_object_classes = 4
        self.hand_pose_classes = 3
        self.x_translation = 0 # dynamically calculated based on number of frames in buffer
        self.translation_factor = 1.5

        self.overlap_thresh = 0.1
        self.hand_area_thresh = 3700
        self.num_rollback_frames = 5

        self.hand_pose_class_map = {0: "None", 1: "PICK", 2:"NOT_PICK"}        
        self.object_class_map = {1: 'Can', 2: 'Bottle', 3: 'Milk Jug', 4: 'Cardboard'}
        self.object_score_threshold_dict =  {1: 0.5126, 2: 0.8644, 3: 0.1859, 4: 0.1859}

        self.buffer_size = 13
        self.rgb_image_buffer = [] 
        self.pick_classification_buffer = []
        self.lm_buffer = [] 
        self.frame_timestamp_buffer = []
        self.image_counter_buffer = []

        self.pick_count = 0
        self.kp_count = 0
        self.image_counter = 0
        self.not_pick_count = 0
        self.crop_count = 0

        # Initialize models
        self.hand_detector = HandDetector()
        self.hand_pose_classifier = ImageClassifier()
        self.hand_pose_classifier.load_state_dict(torch.load(self.hand_pose_model_path))
        self.hand_pose_classifier = self.hand_pose_classifier.to(self.device)
        self.object_detect_model = init_detector(self.config_path, self.object_detect_model_path, device=self.device)

        # Publishers
        self.hand_detection_image_pub = rospy.Publisher('/recycling_stretch/hand_detection_image', Image, queue_size=5)
        self.hand_pose_prediction_image_pub = rospy.Publisher('/recycling_stretch/hand_pose_prediction_image', Image, queue_size=5) # KK: Shows feed whenever a pick is detected
        self.object_detection_image_pub =  rospy.Publisher('/recycling_stretch/detection_image/picked_object', Image, queue_size=5) #KK: Feed with all the objects on belt with green boudning boxes from first frame in queque + red bounding box over hand in real time.  used to calculate overlap
        self.picked_object_image_pub =  rospy.Publisher('/recycling_stretch/picked_object', Image, queue_size=5) # Feed with bounding boxes over objects       
        # self.hand_landmark_pub = rospy.Publisher('/recycling_stretch/hand_landmark', HandLandmark, queue_size=10)
        self.hand_keypoint_pub = rospy.Publisher('/recycling_stretch/hand_landmark', Image, queue_size=10)
        # self.hand_keypoint_pub = rospy.Publisher('/recycling_stretch/hand_landmark', T, queue_size=10)

            #  KK: only significant publishers, publishers. puvs above for visualization
        self.positive_crops_pub = rospy.Publisher('/recycling_stretch/positive_crops', Image, queue_size=5) #KK: Image of item picked cropped
        self.positive_crop_properties_pub = rospy.Publisher('/recycling_stretch/positive_crop_properties', PositiveCropProperties, queue_size=10)
        self.negative_crops_pub = rospy.Publisher('/recycling_stretch/negative_crops', NegativeCrops, queue_size=10)
        self.negative_crop_properties_pub = rospy.Publisher('/recycling_stretch/negative_crop_properties', NegativeCropProperties, queue_size=10)
       
        # Subscribers

        #self.camera_info_sub = message_filters.Subscriber('/camera3/color/camera_info', CameraInfo)
        self.rgb_img_sub = message_filters.Subscriber('/camera3/color/image_raw/compressed', CompressedImage)
        #self.depth_img_sub = message_filters.Subscriber('/camera3/aligned_depth_to_color/image_raw', Image)


        self.synchronizer = message_filters.TimeSynchronizer(
            #[self.rgb_img_sub, self.depth_img_sub, self.camera_info_sub], 10)
            [self.rgb_img_sub], 10)
        self.synchronizer.registerCallback(self.get_image_cb)

        # main thread just waits now..
        rospy.spin()

    def camera_info_cb(self, msg):
        print("getting camera info")
        print(msg)

    def get_image_cb(self, ros_rgb_image):

        self.image_counter+=1

        # self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.rgb_image = self.convert_ros_compressed_to_cv2(ros_rgb_image)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp

        #self.depth_image = ros_numpy.numpify(ros_depth_image)
        #self.depth_image_timestamp = ros_depth_image.header.stamp
 
        # OpenCV expects bgr images, but numpify by default returns rgb images.
        # self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        '''
        time_diff = self.rgb_image_timestamp - self.depth_image_timestamp
        time_diff = abs(time_diff.to_sec())
        if time_diff > 0.0001:
            print('WARNING: The rgb image and the depth image were not taken at the same time.')
            print('         The time difference between their timestamps =', time_diff, 's')
        '''
        
        hand_detect_img = self.hand_detector.findHands(self.rgb_image) # viz
        lmlist = self.hand_detector.findPosition(hand_detect_img)
 
        # Pass img to run inference on the PICK classifier
        pick_pred = self.pick_classifier(hand_detect_img) # viz KK: integer here reprensents labels for pick classifier{0: "None", 1: "PICK", 2:"NOT_PICK"}        
        pick_pred = pick_pred.cpu().detach().numpy()
        pick_class = self.hand_pose_class_map[pick_pred[0]]
        # pick_image = cv2.putText(self.rgb_image, pick_class, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

        '''
        if pick_class == "PICK":
            print(pick_class, self.pick_count)
        else:
            print(pick_class, self.not_pick_count)
        '''
        # KK: creates a queue of frames whenever its not pick
        if pick_class!='PICK':  
            # Buffer output of classifier (PICK/NOT_PICK/None)

            self.rgb_image_buffer = self.build_buffer(self.rgb_image_buffer, hand_detect_img)
            self.image_counter_buffer = self.build_buffer(self.image_counter_buffer, self.image_counter)
            # cv2.imwrite("buffer.jpg", self.rgb_image_buffer[0])
            # print(len(self.rgb_image_buffer), "image buffer")
            self.not_pick_count += 1

            
            # Reset not_pick sequence saved in buffer post pick
            if self.pick_count>=5:
                # print("here")
                if self.not_pick_count > self.buffer_size:
                    # pdb.set_trace()
                    self.rgb_image_buffer = self.rgb_image_buffer[-self.buffer_size:]
                    self.image_counter_buffer = self.image_counter_buffer[-self.buffer_size:] # KK: Queue of frame numbers, unnecessary because build buffer already handles this
                    self.not_pick_count = self.buffer_size
                    # print(self.image_counter_buffer) 
                    
                    # self.rgb_image_buffer = []
                    # self.image_counter_buffer = []
                    # self.not_pick_count = 7
                    self.pick_count = 0
                    # TODO: Trace out logic for resetting pick_count and behavior for false not_pick frames in pick portion of sequence
            # else:
                # print("PICK COUNT < 5: ", self.pick_count)
                    
            if self.not_pick_count <= 3: #TODO: Check if this is unnecessary because of return line after
                return
            return

        # Draw a rotated bounding box on the hand based on the contents of the lmlist at the frame where onset of PICK is detected and all the subsequent
        # frames classified as PICK
        # print(len(lmlist), len(self.rgb_image_buffer))

        # KK: When hand is detected and some frames have been saved 
        if len(lmlist) != 0 and len(self.rgb_image_buffer)>0:
            # print("update pick count") 
            self.pick_count +=1
            self.kp_count += 1
            # hand_box, hand_box_viz = self.draw_box_over_keypoints(lmlist, hand_detect_img) # viz
             # draw translated hand box
            hand_box, hand_box_viz = self.draw_box_over_keypoints_translated(lmlist, hand_detect_img) # viz

            
            # Visualize hand pose prediction
            if VIZ:
                self.ann_image = self.visualize_pick_predictions(hand_box_viz, pick_class)
                # cv2.imwrite("ann_image.jpg", self.ann_image)
                try:
                    pick_image_msg = self.cv2_to_imgmsg(self.ann_image)
                except CvBridgeError as e:
                    rospy.logerr(e)
                    return
                pick_image_msg.header = ros_rgb_image.header
                self.hand_pose_prediction_image_pub.publish(pick_image_msg)
                    
            # Run the object detector on the image obtained in the previous step and obtain rotated bounding boxes on all objects in the belt
            detection_bboxes, detection_classes, detection_scores, detection_rboxes = self.detect_object(self.rgb_image_buffer[0]) # viz


            if VIZ:
                # detection_image = self.visualize_object_bboxes(self.rgb_image_buffer[0], detection_bboxes, detection_classes, detection_scores, detection_rboxes)
                detection_image = self.visualize_object_bboxes(self.rgb_image, detection_bboxes, detection_classes, detection_scores, detection_rboxes) # hand_box_viz
                if detection_image is not None:
                    try:
                        detection_image_msg = self.cv2_to_imgmsg(detection_image)
                    except CvBridgeError as e:
                        rospy.logerr(e)
                        return

                    detection_image_msg.header = ros_rgb_image.header
                    self.object_detection_image_pub.publish(detection_image_msg)
                else:
                    print("No Object!!!")

            # Find the intersection between a future position of the bounding box (translated along the x axis by x amount) to the bounding box drawn on the hand
            self.x_translation = self.translation_factor * (self.image_counter - self.image_counter_buffer[0])            
            translated_hand_rboxes = self.translate_hand_bboxes(hand_box)
            
            # false hand detection
            if(Polygon(translated_hand_rboxes[0]).area < self.hand_area_thresh): #TODO: Add upper bound limit as well for landmarks registered on carboard
                print("False Hand Detected")
                return


            max_overlap = -1
            pos_box_id = -1
            neg_box_ids = []
            for i, box_detect in enumerate(detection_rboxes):
                # compute overlap between rotated bounding box and hand
               
                # pdb.set_trace()
                overlap = self.compute_overlap(Polygon(translated_hand_rboxes[0]), Polygon(box_detect))
                # print("Polygon properties", i , Polygon(box_detect))       
                object_box_centroid = list(Polygon(box_detect).centroid.coords)[0]
                hand_box_viz = cv2.putText(hand_box_viz, str(round(overlap, 2)), (int(object_box_centroid[0]), int(object_box_centroid[1])) , cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 255, 0), 5)
                print(i, overlap, "############################################")
                # If the intersection is above a certain threshold, save the crop of the bounding box to disk along with the predicted category of the object and the prediction confidence
                if overlap >= self.overlap_thresh:
                    if overlap > max_overlap:
                        pos_box_id = i
                        max_overlap = overlap
                else:
                    neg_box_ids.append(i)
                # pdb.set_trace()
            # print(pos_box_id, neg_box_ids) 


           
            if VIZ:
                # detection_image = self.visualize_object_bboxes(self.rgb_image_buffer[0], detection_bboxes, detection_classes, detection_scores, detection_rboxes)
                
                detection_image = self.visualize_object_bboxes(hand_box_viz, detection_bboxes, detection_classes, detection_scores, detection_rboxes)
                if detection_image is not None:
                    try:
                        detection_image_msg = self.cv2_to_imgmsg(detection_image)
                    except CvBridgeError as e:
                        rospy.logerr(e)
                        return

                    detection_image_msg.header = ros_rgb_image.header
                    self.object_detection_image_pub.publish(detection_image_msg)
                else:
                    print("No Object!!!")

            ################### Positive Crops ##############################################

            if pos_box_id == -1:
                return
            box_detect = detection_rboxes[pos_box_id]
            # print(detection_classes[pos_box_id]) 

            pos_crop_properties_msg = PositiveCropProperties()
            pos_crop_properties_msg.header = ros_rgb_image.header
            pos_crop_properties_msg.pred_category = int(detection_classes[pos_box_id])
            pos_crop_properties_msg.pred_confidence = detection_scores[pos_box_id]
            self.positive_crop_properties_pub.publish(pos_crop_properties_msg)

            pos_crop = self.extract_rotated_crop(self.rgb_image_buffer[0], box_detect, detection_classes[pos_box_id], "crop_images_overlap_2/crop_"+str(pos_box_id)+"_"+str(self.crop_count)+"_"+str(overlap)+".jpg")             
            print("SAVED!!!!!")
            self.crop_count += 1
            try:
                crop_image_msg = self.cv2_to_imgmsg(pos_crop)
            except CvBridgeError as e:
                rospy.logerr(e)
                return

            crop_image_msg.header = ros_rgb_image.header
            self.positive_crops_pub.publish(crop_image_msg)
            # cv2.imwrite("pos.png", pos_crop)
            # cv2.imwrite("image.png", self.rgb_image_buffer[0])

            ################### Negative Crops ##################################################

            neg_crops = []
            neg_pred_categories = []
            neg_pred_confidences = []
            for neg_box_id in neg_box_ids:
                neg_crop = self.extract_rotated_crop(self.rgb_image_buffer[0], detection_rboxes[neg_box_id], detection_classes[neg_box_id], "crop_images_overlap_2/neg_"+str(self.crop_count)+"_"+str(max_overlap)+".jpg")
                # cv2.imwrite("neg_"+str(neg_box_id)+'.png', neg_crop)
                try:
                    neg_crop_image_msg = self.cv2_to_imgmsg(neg_crop)
                except CvBridgeError as e:
                    rospy.logerr(e)
                    return                
                neg_crops.append(neg_crop_image_msg)
                neg_pred_categories.append(int(detection_classes[neg_box_id]))
                neg_pred_confidences.append(detection_scores[neg_box_id])
            
            neg_crop_properties_msg = NegativeCropProperties()
            neg_crop_properties_msg.header = ros_rgb_image.header
            neg_crop_properties_msg.pred_category = neg_pred_categories
            neg_crop_properties_msg.pred_confidence = neg_pred_confidences
            self.negative_crop_properties_pub.publish(neg_crop_properties_msg)

            neg_crops_msg = NegativeCrops()
            neg_crops_msg.header = ros_rgb_image.header
            neg_crops_msg.data = neg_crops
            self.negative_crops_pub.publish(neg_crops_msg)
            
    def pick_classifier(self, image):
        '''
        Wrappper over the pick classifier. Manipulates the images in the same way as the train dataloader and 
        runs inference on the trained model

        Parameters:
                    image (image): Input image from get_image_cb
        Returns:
                    predicted (int): Predicted class 0/1/2 (None/PICK/NOT_PICK)
        '''
        device = self.device
        with torch.no_grad():
            image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_CUBIC)
            image = np.asarray(image, np.float32)
            image -= self.mean
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, axis=0)
            image = torch.tensor(image)
            outputs = self.hand_pose_classifier(image.to(device))
            outputs = outputs.to(device)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def detect_object(self,image):
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
        all_bboxes, all_classes, all_scores, all_rboxes = run_inference(self.object_detect_model, image, self.num_object_classes)
        return all_bboxes, all_classes, all_scores, all_rboxes

    def build_buffer(self, buffer, buffer_contents):
        '''
        Helper function to populate a buffer with data

        Parameters:
                    buffer (list): Empty list 
                    buffer_contents (list): List of data (image/int/float)
        Returns:
                    buffer (list): Queue of items from the buffer_contents
        '''
        buffer.append(buffer_contents)
        if len(buffer) > self.buffer_size:
            buffer.pop((0))
        return buffer

    def translate_hand_bboxes(self, box):
        '''
        Helper function to translate the bounding box drawn over the hands by joining the keypoints a certain distance

        Parameters:
                    box (list(list)): Rotated bounding box obtained by joining the exterior of the predicted hand keypoints in the 
                                      format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        Returns:
                    rbox_list (list(list(list))): Bounding box translated by self.x_translation
        '''
        rbox_list = []
        x1, y1 = box[0][0], box[0][1]
        x2, y2 = box[1][0], box[1][1]
        x3, y3 = box[2][0], box[2][1]
        x4, y4 = box[3][0], box[3][1]
        x1_t = x1 - self.x_translation
        x2_t = x2 - self.x_translation
        x3_t = x3 - self.x_translation
        x4_t = x4 - self.x_translation
        rbox_list.append([[x1_t, y1], [x2_t, y2], [x3_t, y3], [x4_t, y4], [x1_t, y1]])
        return rbox_list 

    def translate_bboxes(self, detection_rboxes):
        '''
        Helper function to translate the detected bounding boxes drawn over the objects a certain distance

        Parameters:
                    detection_rboxes (list(list)): List of rotated bounding boxes for all the detected objects in the 
                                      format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        Returns:
                    rbox_list (list(list(list))): Bounding boxes translated by self.x_translation
        '''
        rbox_list = []
        for box in detection_rboxes:
            x1, y1 = box[0][0], box[0][1]
            x2, y2 = box[1][0], box[1][1]
            x3, y3 = box[2][0], box[2][1]
            x4, y4 = box[3][0], box[3][1]
            x1_t = x1 + self.x_translation
            x2_t = x2 + self.x_translation
            x3_t = x3 + self.x_translation
            x4_t = x4 + self.x_translation
            rbox_list.append([[x1_t, y1], [x2_t, y2], [x3_t, y3], [x4_t, y4], [x1_t, y1]])
        return rbox_list

    def compute_overlap(self, box1, box2):
        '''
        Helper function to calculate the overlap between two polygons

        Parameters:
                    box1 (Polygon): Bounding box 1 converted to Polygon
                    box2 (Polygon): Bounding box 2 converted to Polygon
        Returns:
                    overlap (float): Overlap between the bounding boxes
        '''
        try:
            intersect_area = float(box1.intersection(box2).area) 
            if intersect_area == 0.0:
                return 0.0
            overlap = intersect_area / (box1.area + box2.area - intersect_area)
            return overlap
        except Exception as e:
            print(box1.area, box2.area, intersect_area)
            print(e)
            print(box1, box2)
            return 0.0

    def draw_box_over_keypoints(self, lm, img):
        '''
        Helper function to draw a tightly fitted box by joining the predicted keypoints on the hand

        Parameters:
                    lm (list): List of landmarks predicted on the hand in the format [x0, y0, x1, y1, ..., x20, y20]
                    img (image): Input image from get_image_cb
        Returns:
                    box (list(list)): Bounding box over the hand in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    result (image): Image with the bounding box drawn over the hand
        '''
        h, w, c = img.shape
        coord_list = []

        # Parse landmarks and get them in an appropriate format [[x0, y0], [x1, y1], ..... [x20, y20]]
        for i in range(0, len(lm), 2):
            coord = lm[i:i+2]
            # print(coord)
            for c in coord:
                x = c[1]
                y = c[2]
                if x!='None' and y!='None':
                    coord_list.append([int(x), int(y)])
                    cx, cy = int(x), int(y) 
                    img = cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        coord_list = np.array(coord_list)

        # Read coord_list and draw a tightly fitted bounding box
        if len(coord_list) > 0:
            rect = cv2.minAreaRect(coord_list)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            

            result = img.copy()
            # cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
            return box, result
    
    # TODO: find more elegant sol -> maybe add param to orig func
    def draw_box_over_keypoints_translated(self, lm, img):
        '''
        Helper function to draw a tightly fitted box by joining the predicted keypoints on the hand

        Parameters:
                    lm (list): List of landmarks predicted on the hand in the format [x0, y0, x1, y1, ..., x20, y20]
                    img (image): Input image from get_image_cb
        Returns:
                    box (list(list)): Bounding box over the hand in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    result (image): Image with the bounding box drawn over the hand
        '''
        h, w, c = img.shape
        coord_list = []

        # Parse landmarks and get them in an appropriate format [[x0, y0], [x1, y1], ..... [x20, y20]]
        for i in range(0, len(lm), 2):
            coord = lm[i:i+2]
            # print(coord)
            for c in coord:
                x = c[1]
                y = c[2]
                if x!='None' and y!='None':
                    coord_list.append([int(x), int(y)])
                    cx, cy = int(x), int(y) 
                    img = cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        coord_list = np.array(coord_list)

        # Read coord_list and draw a tightly fitted bounding box
        if len(coord_list) > 0:
            rect = cv2.minAreaRect(coord_list)
            box = cv2.boxPoints(rect)
            print("box", box)
            # Translate boxes
            for b in box:
                # print("b", b)
                # print("Old x_pos:", box[0][0],"\nNew x_pos:", box[0][0] - self.x_translation )
                b[0] = b[0] - self.x_translation
                print("TRANSLATING")
            box = np.int0(box)         

            result = img.copy()
            # cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
            return box, result


    def extract_rotated_crop(self, img, box, cat, img_name):
        '''
        Helper function to extract the crop of a rotated bounding box by applying some image transformations
        https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/

        Parameters:
                    img (image): Input image from get_image_cb
                    box (list(list)): Rotated bounding box over an object in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    cat (int): Predicted category of the object
                    img_name (string): Name to save the crop on disk
        Returns:
                    warped (image): Extracted crop
        '''
        cnt = np.array(box)
        rect = cv2.minAreaRect(cnt)

        # the order of the box points: bottom left, top left, top right, bottom right
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")

        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))
        # cv2.putText(warped, str(cat), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        cv2.imwrite(img_name, warped)
        return warped
    
    def visualize_pick_predictions(self, image, pick_pred):
        '''
        Visualize the predictions of the Pick classifier on an image

        Parameters:
                    image (image): Input image from get_image_cb
                    pick_pred (int): Predicted class 0/1/2 (None/PICK/NOT_PICK)
        Returns:
                    ann_image (image): Image with the prediction visualized on the top left corner of the image
        '''
        # ann_image = cv2.putText(image, pick_pred, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        ann_image = image
        return ann_image
        # return image
        
    def visualize_object_bboxes(self, detection_image, detection_bboxes, detection_classes, detection_scores, detection_rboxes):
        '''
        Vizualize rotated bounding boxes on the image

        Parameters:
                    detection_boxes (list(list)): List of all axis aligned bounding boxes in the [(x_min, y_min), (x_max, y_max)] format
                    detection_classes (list): List of all the predicted classes
                    detection_scores (list): List of all confidence scores for all the predicted boxes
                    detection_rboxes (list(list)): List of coordinates for rotated bounding boxes derived from predicted masks
                                                in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)] format
        Returns:
                    detection_image (image): Image with bounding boxes drawn on the image
        '''
        # detection_image = self.rgb_image 
        for i, _ in enumerate(detection_scores):
            score = detection_scores[i]
            class_id = detection_classes[i]
            threshold = self.object_score_threshold_dict[class_id]
            score = np.round(score, 2)

            if score < threshold:
                continue

            detection_image = self.draw_rboxes(self.ann_image, detection_rboxes[i], class_id, COLOR, BBOX_THICKNESS, score, put_score= False)
        return detection_image
    
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
        '''
        if put_text:
            cv2.putText(image, self.object_class_map[class_id],(rboxes[0][0],rboxes[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        if put_score:
            cv2.putText(image,str(score),(rboxes[1][0],rboxes[1][1]),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        '''
        return image

    def convert_ros_compressed_to_cv2(self, compressed_msg):
        np_arr = np.fromstring(compressed_msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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
        node = FindPickedObject()
    except rospy.ROSInterruptException:
        pass
