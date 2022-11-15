#!/usr/bin/env python

from email import message
import cv2
from cv2 import cvtColor
import numpy as np
# from recycling_stretch.scripts.real_time_object_detection import DRAW_AA_BBOX, DRAW_RBBOX
import rospy
from recycling_stretch.msg import DebugDetectedBox, PositiveCropProperties, NegativeCropProperties, NegativeCrops, SimDetectedBox, DebugSimDetectedBox
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image 
import message_filters
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Int64MultiArray, Float64MultiArray, MultiArrayDimension, Int16
import ros_numpy
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image as ImageShow
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import softmax
from shapely.geometry import Polygon, Point
import sys
from itertools import cycle
import random

# contrastive clustering imports
from human_supervised_cc.modules import transform, resnet, network, contrastive_loss


VIZ = True
SAVE_VIZ = False
DRAW_AA_BBOX = False
DRAW_RBBOX = True
BBOX_THICKNESS = 2
COLOR = (0, 255, 0)
EPOCH_OF_INTEREST = 600

class InferenceContrastiveSimilarity():
    def __init__(self):

        # Init the node
        rospy.init_node('contrastive_similarity_inference')

        # Node parameters
        self.target_subcat = 'Trays'
        self.epoch = 100
        self.device = 'cuda'
        self.resnet = 'ResNet18'
        self.projection_dim = 128
        self.temperature = 0.5
        self.pos_sim_threshold = 0.1
        self.mean = (128, 128, 128)
        self.sup = False
        self.epoch_num = 1000
        self.count = 0
        self.online_training_weights_path = os.path.join(os.path.expanduser("~"), "contrastive_training_weights", self.target_subcat)
        
        self.online = True # compare with online data
        self.online_eval = True # evaluate using weights saved from contrastive_similarity_training
        self.human_crop_list = []
        self.temp_human_crop_list = []
        self.score_threshold_dict =  {1: 0.5126, 2: 0.8644, 3: 0.1859, 4: 0.1859} 
        self.num_human_crops = 40
        self.dominant_cat = None
        self.fresh_start = False # new model
        
        self.class_num = 11

        # Read offline human list
        self.data_root = os.path.join(os.path.expanduser("~"), "Crops_Dataset")
        self.list_id = 2
        self.file_name = "Human_" + self.target_subcat + "_" + str(self.list_id) + ".txt"
        self.file_path = os.path.join("file_lists/human_lists/", self.file_name)        
        self.human_txt_file_root = os.path.join(self.data_root, self.file_path)

        # Initialize models
        self.cc_model_path = os.path.join(os.path.expanduser('~'), 
                                        "catkin_ws/src/recycling_stretch/scripts/contrastive_clustering_occ/save/crops")
        self.encoder = resnet.get_resnet(self.resnet)

        self.cc_model = network.Network(self.encoder, self.projection_dim, self.class_num)
        model_fp = os.path.join(self.cc_model_path, "checkpoint_{}.tar".format(self.epoch_num))
        self.cc_model.load_state_dict(torch.load(model_fp, map_location=self.device)['net'])
        print("Loaded weights!", model_fp)

        self.cc_model = self.cc_model.to(self.device)
        self.cc_model.eval()

        # Publishers
        self.sim_detected_box_pub = rospy.Publisher('/recycling_stretch/sim_detection_boxes', DebugSimDetectedBox, queue_size=10)

        self.sim_detection_image_pub = rospy.Publisher('/recycling_stretch/sim_detection_image', Image, queue_size=10)
       
        # Subscribers
        self.positive_crops_sub = message_filters.Subscriber('/recycling_stretch/positive_crops', Image)
        self.positive_crop_properties_sub = message_filters.Subscriber('/recycling_stretch/positive_crop_properties', PositiveCropProperties)
            
        self.crops_synchronizer = message_filters.TimeSynchronizer(
            [self.positive_crops_sub, self.positive_crop_properties_sub], 10)
        self.crops_synchronizer.registerCallback(self.crops_cb)

        self.rgb_img_sub = message_filters.Subscriber('/camera2/color/image_raw', Image)
        self.bbox_pos_img_sub = message_filters.Subscriber('/recycling_stretch/detection_boxes', DebugDetectedBox)
        
        self.synchronizer = message_filters.TimeSynchronizer(
            [self.rgb_img_sub, self.bbox_pos_img_sub], 50)
        
        self.synchronizer.registerCallback(self.get_image_cb)
        
        # main thread just waits now..
        rospy.spin()

           
    def crops_cb(self, pos_crop, pos_crop_properties): # pos_crop = human crop
        '''
        Callback function that receives the crops of objects picked by the human and the properties of those crops, 
        saves the crops in a list and converts the list to a tensor

        Parameters:
                    pos_crop (msg): Message of type Image which is a crop of size (128, 128, 3)
                    pos_crop_properties (msg): Message of type PositiveCropProperties containing information
                                           about the predicted category of the object and the prediction confidence
        '''
        print("New Positive Crop!!")
        pos_crop = self.imgmsg_to_cv2(pos_crop)
        pos_crop = cv2.resize(pos_crop, (128, 128), interpolation=cv2.INTER_CUBIC)
        pos_crop = np.asarray(pos_crop, np.float32)
        pos_crop = pos_crop.transpose((2, 0, 1))
        self.temp_human_crop_list.append(pos_crop)
        self.human_crop_list = torch.as_tensor(self.temp_human_crop_list)

    def get_image_cb(self, ros_rgb_image, bbox_pos):
        '''
        Separate Callback function that receives the info from predicted bounding boxes, runs inference on the 
        image stream, and returns the similarity for each object on the belt

        Parameters:
                    ros_rgb_image (msg): Message of type Image, which is the real time feed from camera 2
                    bbox_pos (msg): Message of type DetectedBox which is the contains the coordinates and properties of each predicted bounding box

        '''
        self.rgb_image = ros_numpy.numpify(ros_rgb_image)
        self.rgb_image_timestamp = ros_rgb_image.header.stamp

        # OpenCV expects bgr images, but numpify by default returns rgb images.
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

        # Convert rgb image message back to cv2
        cv_image = self.imgmsg_to_cv2(ros_rgb_image)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        # Output of the object detector
        rboxes_list, cat_list = self.process_rbboxes(bbox_pos)

        similarity_list = []

        # Make human crops list offline for debugging and benchmarking 
        # Set self.online = False when testing the real system
        if not self.online:
            print("Comparing with offline data")
            # Read predefined human selected crops from a text file
            self.human_crop_list = self.offline_read_human_crop()
            self.human_crop_list = torch.as_tensor(self.human_crop_list)
            self.human_crop_list = torch.permute(self.human_crop_list, (0, 3, 1, 2))
        else:
            print("Comparing with online data!!")
        
        self.count = self.count + 1

        # When starting with a new category
        if self.fresh_start:
            print("Fresh Start")
            self.fresh_start = False
    
        if self.online_eval:
            print("Loading Latest Weights")
            self.read_latest_weights_for_inference(EPOCH_OF_INTEREST)

        # iterate over all boxes
        if len(self.human_crop_list) > 0:
            for i, (box, cat) in enumerate(zip(rboxes_list, cat_list)):

                # get reference crop by indexing the box into the image and warping it to form a usable crop
                ref_crop = self.extract_rotated_crop(cv_image, box)

                # process the ref crop for pytorch evaluation
                ref_crop = self.image_manipulation(ref_crop.copy())
    
                # for each ref_crop, calculate similarity to each human crop
                # sim[0] = [ref_crop, human_1, human_2, ......., human_n]

                # If too many items have been picked, compare with a fixed set of items to deal with compute bottleneck
                # TODO: Figure out a better way of doing this because it reduces the performance
                if len(self.human_crop_list) > self.num_human_crops:
                    print("Comparing with a random selection of positive crops")
                    #random_indices = np.random.choice(len(self.human_crop_list), self.num_human_crops)
                    self.human_crop_list = self.human_crop_list[-self.num_human_crops:]

                # Inference loop
                if len(self.human_crop_list) > 0:
                    # send each box one by one to the inference function as reference crop to compare with the human selected crops
                    similarity = self.inference(ref_crop, self.human_crop_list)

                    # take max of sims from human_1 to human_n (or top5 or some other heuristic) - return one float similarity value
                    sim_row = similarity[0]
                    sims = sim_row[1:].cpu().detach().numpy()
                    max_sims = round(np.max(sims), 3)

                    # save visualizations to file for debugging purposes (slows down real-time performance)
                    if SAVE_VIZ:
                        save_filename = "RealTime_viz/sim_"+ str(self.count)+"_"+str(i)+".jpg"
                        viz = self.visualize_similarity(sim_row, ref_crop, self.human_crop_list, save_filename, max_sims)

                    # store all the similarities in a list
                    similarity_list.append(max_sims)
        else:
            print("Not running inference because human crops list is empty")

        # publish the similarity list and pass it on with the bboxes message
        sim_detected_box = DebugSimDetectedBox()
        sim_detected_box.header = bbox_pos.header
        sim_detected_box.bboxes = bbox_pos.bboxes
        sim_detected_box.rbboxes = bbox_pos.rbboxes
        sim_detected_box.pred_category = bbox_pos.pred_category
        sim_detected_box.scores = bbox_pos.scores
        sim_detected_box.similarity = similarity_list
        # publish similarity values on each bbox
        if VIZ:
            detection_image = self.visualize_bboxes(bbox_pos.bboxes, bbox_pos.pred_category, bbox_pos.scores, bbox_pos.rbboxes, similarity_list)

            if detection_image is not None:
                try:
                    detection_image_msg = self.cv2_to_imgmsg(detection_image)
                except CvBridgeError as e:
                    rospy.logerr(e)
                    return

                detection_image_msg.header = bbox_pos.header
                self.sim_detection_image_pub.publish(detection_image_msg)
                sim_detected_box.img = detection_image_msg
                self.sim_detected_box_pub.publish(sim_detected_box)
            else:
                print("No Object!!!")

    def inference(self, pos_data, neg_data):
        '''
        Run inference through a trained SimCLR model

        Parameters:
                    pos_data (tensor): Image converted to torch tensor of shape (1, 3, 128, 128)
                    neg_data (tensor): Tensor containing list of negative images of shape (n, 3, 128, 128)
        Returns:
                    sim (tensor): A tensor of size (n+1 x n+1) containing values between 0 and 1 denoting pairwaise similarity between each item
        '''
        print("Running Inference!")
        similarity_f = nn.CosineSimilarity(dim=2)
        with torch.no_grad():
            x_i = pos_data.to(self.device) # reference crop
            x_j = neg_data.to(self.device) # human-selected crops
            x = torch.cat((x_i, x_j))
            # the paper takes the representations of the encoder and not the MLP during evaluation
            h, _, _, _, _, _, _ = self.cc_model(x, x, x) 
            sim = similarity_f(h.unsqueeze(1), h.unsqueeze(0))
            return sim

    def process_rbboxes(self, bbox_pos):
        '''
        Process data from the bounding box topic to convert it to a usable format for running inference

        Patameters:
                    bbox_pos(msg): Message in the format DetectedBox
        Returns:
                    box_list(list): List of rotated bounding boxes in a frame in the format 
                                    [[list(top_left), list(top_right), list(bottom_left), list(bottom_right), list(top_left)], 
                                     [list(top_left), list(top_right), list(bottom_left), list(bottom_right), list(top_left)],
                                     ....]
                    cat_list(list): List of predicted categories of each bounding box
        '''
        rboxes = bbox_pos.rbboxes
        categories = bbox_pos.pred_category
        box_list = []
        cat_list = []
        for (rbox, cat) in zip(rboxes, categories):
            top_left = [int(rbox.top_left_x), int(rbox.top_left_y)]
            top_right = [int(rbox.top_right_x), int(rbox.top_right_y)]
            bottom_left = [int(rbox.bottom_left_x), int(rbox.bottom_left_y)]
            bottom_right = [int(rbox.bottom_right_x), int(rbox.bottom_right_y)]
            box = [list(top_left), list(top_right), list(bottom_left), list(bottom_right), list(top_left)]
            box_list.append(box)
            cat_list.append(cat)
        return box_list, cat_list

    def image_manipulation(self, image):
        '''
        Manipulate a croped image to get it ready for inference through the SimCLR model
        Do all the operations mimicking the offline PyTorch dataloader

        Parameters:
                    image(np array): Extracted croped image using the predicted rotated bounding box of arbitrary size
        Returns:
                    image(tensor): Croped image converted to tensor of shape (1, 3, 128, 128)
        '''
        # mean subtraction from image
        # image -= self.mean
        # transpose to get channels first
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image)
        image = image.unsqueeze(0)
        return image
    
    def extract_rotated_crop(self, img, box):
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
        cnt = np.array(list(box))
        rect = cv2.minAreaRect(cnt) ################## CHECK THIS

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
        # if np.shape(warped)[0] > 100 and np.shape(warped)[1] > 100:
        # cv2.imwrite(img_name, warped)
        return warped

    #################################################
    # Reading Data from Disk Utilities
    #################################################  

    def read_latest_weights_for_inference(self, latest_epoch):
        '''
        When running inference while online training is happening, look for the latest available trained weights 
        and load that to run inference
        '''
        available_weights = []
        if os.path.exists(self.online_training_weights_path):
            for weight_name in os.listdir(self.online_training_weights_path):
                epoch = weight_name.split('_')[1].split('.')[0] # 2
                available_weights.append(int(epoch))
            available_weights = sorted(available_weights)
            # latest_epoch = available_weights[-1]
            online_model_path = os.path.join(self.online_training_weights_path, self.target_subcat+"_"+str(latest_epoch)+".pth")
            self.cc_model.load_state_dict(torch.load(online_model_path))
            print("LOADED LATEST WEIGHTS FROM EPOCH: " + str(latest_epoch))

    def offline_read_human_crop(self):
        '''
        When evaluating against pre-saved human crops from list, read the list and convert each image in the list to a usable format
        for inference

        Returns:
                human_image_list(list): List of cropped images, where each crop is of size (128, 128, 3)
        '''
        
        human_data = np.loadtxt(self.human_txt_file_root, dtype=str)
        human_image_list = []
        human_name_list = []
        for i, cd in enumerate(human_data):
            human_name_list.append(cd)
            image = ImageShow.open(cd)
            image = image.resize((128, 128))
            image = np.asarray(image, np.float32)
            human_image_list.append(image)
        return human_image_list

    #################################################
    # Visualization Message Utilities
    ################################################# 

    def process_output_for_viz(self, crop_tensor):
        '''
        Convert image tensor back to numpy for visualization

        Parameters:
                    crop_tensor (tensor): Tensor containing image of size (3, 128, 128)
        Returns:
                    crop (np array): Numpy Array for visualization of size (128, 128, 3)

        '''
        crop = crop_tensor.detach().cpu().numpy()
        crop = crop.transpose((1, 2, 0))
        crop = np.asarray(crop, np.uint8)
        # crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        return crop   
    
    def visualize_similarity(self, sim_row, pos_crop, neg_crop, save_filename, max_sims):
        '''
        Visualize similarity of reference crop with respect to the human selected crops in a grid and save that to disk

        Parameters:
                    sim_row (list): List of similarities of the reference object with respect to the human selected object 
                                    (first row of the similarity matrix)
                    pos_crop (tensor): Reference crop image of shape (1, 3, 128, 128)
                    neg_crop (tensor): Tensor of list of all human selected crops (n, 3, 128, 128)
                    save_filename (string): Filename for saving the visualizations
                    max_sims (float): Aggregated similarity metric for a given reference crop
        Returns:
                    fig (plt.figure): Object of the plotted figure
        '''
        fig = plt.figure()
        neg_sim = sim_row[1:]
        plt.title("Max: " + str(max_sims), loc='left')
        outer_gs = gridspec.GridSpec(1, 2)
        inner_gs0 = outer_gs[0].subgridspec(1,1)
        axs_1 = fig.add_subplot(inner_gs0[0, 0])
        pos_sim = sim_row[0]
        pos_crop = torch.squeeze(pos_crop, 0)
        pos_crop = self.process_output_for_viz(pos_crop)
        pos_crop = cv2.cvtColor(pos_crop, cv2.COLOR_RGB2BGR)
        axs_1.imshow(pos_crop)
        axs_1.set_axis_off()
        axs_1.set_title(str(round(float(pos_sim.detach().cpu().numpy()), 3)))
                
        N = len(neg_sim)
        cols = 3
        rows = int(math.ceil(N / cols))
        for i, (nc, ns) in enumerate(zip(neg_crop, neg_sim)):
            gs = outer_gs[1].subgridspec(rows, cols)
            ax = fig.add_subplot(gs[i])
            nc = self.process_output_for_viz(nc)
            ax.set_axis_off()
            ax.imshow(nc)
            ax.set_title(str(round(float(ns.detach().cpu().numpy()), 3)))
        plt.axis('off')
        plt.savefig(save_filename)
        print(save_filename)
        return fig
        
    def visualize_bboxes(self, detection_bboxes, detection_classes, detection_scores, detection_rboxes, similarity_list):
        '''
        Vizualize axis-aligned and rotated bounding boxes on the image

        Parameters:
                    detection_boxes (list(list)): List of all axis aligned bounding boxes in the [(x_min, y_min), (x_max, y_max)] format
                    detection_classes (list): List of all the predicted classes
                    detection_scores (list): List of all confidence scores for all the predicted boxes
                    detection_rboxes (list(list)): List of coordinates for rotated bounding boxes derived from predicted masks
                                                   in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)] format
                    similarity_list (list): List of float values between 0 and 1 denoting the similarity
        Returns:
                    detection_image (image): Image with bounding boxes drawn on the image
        '''
        detection_image = self.rgb_image 
        if len(similarity_list) > 0:
            for i, _ in enumerate(detection_scores):
                score = detection_scores[i]
                class_id = detection_classes[i]
                similarity = similarity_list[i]
                threshold = self.score_threshold_dict[class_id]
                score = np.round(score, 2)

                if score < threshold:
                    continue

                if DRAW_AA_BBOX:
                    detection_image = self.draw_aa_boxes(self.rgb_image, detection_bboxes[i], similarity, COLOR, BBOX_THICKNESS, score, put_text=True, put_score=True)

                if DRAW_RBBOX:
                    detection_image = self.draw_rboxes(self.rgb_image, detection_rboxes[i], similarity, COLOR, BBOX_THICKNESS, score, put_score= False)
            return detection_image
        else:
            print("Cannot visualize!!")
            return detection_image
    
    def draw_aa_boxes(self, image, box, similarity, color, thickness, score, put_text, put_score=True):
        '''
        Helper function to draw axis aligned bounding boxes on the image

        Parameters:
                    image (image): Input image from get_image_cb
                    box (list): Coordinates of a bounding box in the [x_min, y_min, x_max, y_max] format
                    similarity (float): similarity score
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
            cv2.putText(image, str(similarity),(start_point),cv2.FONT_HERSHEY_SIMPLEX,1, color, thickness)
        if put_score:
            cv2.putText(image,str(score),(start_point),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        return image

    def draw_rboxes(self, image, rboxes, similarity, color, thickness, put_text, score=None, put_score=True):
        '''
        Helper function to draw rotated bounding boxes on the image

        Parameters:
                    image (image): Input image from get_image_cb
                    rboxes (RotatedBBox): Coordinates of a bounding box in the [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] format
                    similarity (int): similarity score
                    color (tuple): Color of the bounding box
                    thickness (int): Thickness of the bounding box
                    score (float):  Confidence score of the bounding box
                    put_score (bool): Boolean flag to decide whether to print the score on the bounding box
        Returns:
                    image (image): Image with the bounding boxes plotted on the image
        '''
        
        cv2.line(image, (int(rboxes.top_left_x), int(rboxes.top_left_y)), (int(rboxes.top_right_x), int(rboxes.top_right_y)), color, thickness)
        cv2.line(image, (int(rboxes.top_left_x), int(rboxes.top_left_y)), (int(rboxes.bottom_right_x), int(rboxes.bottom_right_y)), color, thickness)
        cv2.line(image, (int(rboxes.top_right_x), int(rboxes.top_right_y)), (int(rboxes.bottom_left_x), int(rboxes.bottom_left_y)), color, thickness)
        cv2.line(image, (int(rboxes.bottom_right_x), int(rboxes.bottom_right_y)), (int(rboxes.bottom_left_x), int(rboxes.bottom_left_y)), color, thickness)

        if put_text:
            cv2.putText(image, str(similarity),(int(rboxes.top_left_x), int(rboxes.top_left_y)),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        if put_score:
            cv2.putText(image,str(score),(int(rboxes.top_right_x), int(rboxes.top_right_y)),cv2.FONT_HERSHEY_SIMPLEX,1,color, thickness)
        return image
        
    #################################################
    # Image Message Utilities
    #################################################

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
        node = InferenceContrastiveSimilarity()
    except rospy.ROSInterruptException:
        pass
