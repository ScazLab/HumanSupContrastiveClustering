#!/usr/bin/env python

from re import X
from xml import dom
import cv2
import numpy as np
import rospy
from recycling_stretch.msg import DetectedBox, PositiveCropProperties, NegativeCropProperties, NegativeCrops
from cv_bridge import CvBridge, CvBridgeError
from scripts.contrastive_clustering_occ.modules import human_supervised_loss
from sensor_msgs.msg import Image
import message_filters
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Int64MultiArray, Float64MultiArray, MultiArrayDimension, Int16
import ros_numpy
import matplotlib.pyplot as plt
from PIL import Image as ImageShow
import os
import math
import torch
from torch.utils import data
import torch.nn as nn
from shapely.geometry import Polygon, Point
from torch.utils.tensorboard import SummaryWriter
import sys
import random
from operator import itemgetter
from itertools import cycle
from collections import Counter
from torch.nn.functional import softmax
import torchvision

# contrastive clustering imports
from human_supervised_cc.modules import transform, resnet, network, contrastive_loss

class TrainContrastiveSimilarity():
    def __init__(self):

        # Init the node
        rospy.init_node('contrastive_similarity_training')

        # Node parameters
        self.resnet = 'ResNet18'
        self.projection_dim = 128
        self.device = "cuda"
        self.pos_batch_size = 1
        self.neg_batch_size = 32
        self.batch_size = 32
        self.weight_decay = 1e-6
        self.epochs = 100
        self.instance_temperature = 0.5
        self.cluster_temperature = 1.0
        self.world_size = 1
        self.mean = (128, 128, 128)
        self.start_epoch = 0
        self.num_epochs = 5
        self.global_step = 0
        # self.neg_batch_size = 16 # 64
        self.train_sampler = None
        self.optimizer = "Adam"
        self.epoch_num = 1000
        # self.epoch = 100
        self.image_size = (128, 128)
        self.class_num = 11
        self.learning_rate = 0.01

        self.random_data = torch.FloatTensor(np.ones((1, 3, 128, 128)))
        self.start = True
        self.target_subcat = "Tins"
        self.sim_threshold = 0.8

        self.uncertain_negative_threshold = 0.5
        self.uncertain_positive_threshold = 1.1
        self.global_epoch = 0
        self.model_save_dir = os.path.join(os.path.expanduser("~"), "contrastive_training_weights", self.target_subcat)

        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
            print("Created Model Directory: ", self.model_save_dir)
        
        # Data augmentations
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.5 * s, 0.8 * s, 0.2 * s
        )
        gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5))
        random_invert = torchvision.transforms.RandomInvert()
        random_resized_crop = torchvision.transforms.RandomResizedCrop(size=self.image_size)
        random_posterize = torchvision.transforms.RandomPosterize(bits=2)
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([random_resized_crop, # cans
                                                                                        color_jitter, # cans
                                                                                        # random_invert, # cans
                                                                                        gaussian_blur, # cans
                                                                                        # random_posterize # cans
                                                                                        ], 
                                                                                        p=0.9)])

        # Initialize datasets
        self.positive_dataset = PositiveDataSet(self.transforms)
        self.negative_dataset = NegativeDataSet(self.transforms)

        # Initialize positive and negative buffers with random data (to deal with the empty dataloader case)
        self.positive_dataset.pos_image_buffer.append(self.random_data) 
        self.negative_dataset.neg_image_buffer.append(self.random_data)
        
        # self.uncertain_image_buffer = []
        self.temp_negative_buffer = []

        self.positive_categories = np.array([])
        self.negative_categories = np.array([])

        self.positive_confidences = np.array([])
        self.negative_confidences = np.array([])

        # initialize data loader
        self.pos_train_loader = torch.utils.data.DataLoader(self.positive_dataset,
                                                    batch_size=self.pos_batch_size,
                                                    shuffle=(self.train_sampler is None),
                                                    drop_last=True,
                                                    sampler=self.train_sampler,
                                                    )
        self.neg_train_loader = torch.utils.data.DataLoader(self.negative_dataset,
                                                    batch_size=self.neg_batch_size,
                                                    shuffle=(self.train_sampler is None),
                                                    drop_last=True,
                                                    sampler=self.train_sampler,
                                                    )

        ####### Initialize models

        # initialize ResNet
        self.encoder = resnet.get_resnet(self.resnet)

        # initialize model
        self.cc_model_path = os.path.join(os.path.expanduser('~'), 
                                        "catkin_ws/src/recycling_stretch/scripts/contrastive_clustering_occ/save/crops")
        self.encoder = resnet.get_resnet(self.resnet)

        self.cc_model = network.Network(self.encoder, self.projection_dim, self.class_num)
        model_fp = os.path.join(self.cc_model_path, "checkpoint_{}.tar".format(self.epoch_num))
        self.cc_model.load_state_dict(torch.load(model_fp, map_location=self.device)['net'])
        print("Loaded weights!", model_fp)

        self.cc_model = self.cc_model.to(self.device)

        # optimizer / loss
        self.optimizer = torch.optim.Adam(self.cc_model.parameters(), 
                                        lr=self.learning_rate, 
                                        weight_decay=self.weight_decay)

        self.loss_device = torch.device("cuda")
        self.criterion_instance = contrastive_loss.InstanceLoss(self.batch_size+1, 
                                            self.instance_temperature, self.loss_device).to(self.loss_device)
        self.criterion_cluster = contrastive_loss.ClusterLoss(self.class_num, 
                                            self.cluster_temperature, self.loss_device).to(self.loss_device)
        self.criterion_one_class = human_supervised_loss.OneClassLoss(self.loss_device)
        self.writer = SummaryWriter()

        # Publishers
       
        # Subscribers
        self.positive_crops_sub = message_filters.Subscriber('/recycling_stretch/positive_crops', Image)
        self.positive_crop_properties_sub = message_filters.Subscriber('/recycling_stretch/positive_crop_properties', PositiveCropProperties)
        self.negative_crops_sub = message_filters.Subscriber('/recycling_stretch/negative_crops', NegativeCrops)
        self.negative_crops_properties_sub = message_filters.Subscriber('/recycling_stretch/negative_crop_properties', NegativeCropProperties)        

        self.synchronizer = message_filters.TimeSynchronizer(
            [self.positive_crops_sub, self.positive_crop_properties_sub, self.negative_crops_sub, self.negative_crops_properties_sub], 10)
        self.synchronizer.registerCallback(self.crops_cb)

        print("done!")
        # main thread just waits now..
        rospy.spin()
    
    def crops_cb(self, pos_crop, pos_crop_properties, neg_crops, neg_crops_properties):
        '''
        Callback function that receives the positive and negative crops with their propeties, and trains the SimCLR model online

        Parameters:
                    pos_crop (msg): Image message containing positive crops
                    pos_crop_properties (msg): Message of type PositiveCropProperties containing properties of the positive crops
                    neg_crops (msg): List of Image messages containing negative crops
                    neg_crops_properties (msg): Message of type NegativeCropProperties containing properties of the negative crops
        '''
        print("Inside training cb")
        #### Positive Crops
        pos_crop = self.imgmsg_to_cv2(pos_crop)
        pos_crop = self.process_image(pos_crop)

        #### Negative Crops       
        neg_crops_list = []
        for i, (neg_crop, neg_cat) in enumerate(zip(neg_crops.data, neg_crops_properties.pred_category)):
            neg_crop = self.imgmsg_to_cv2(neg_crop)           
            neg_crop = self.process_image(neg_crop.copy())
            neg_crops_list.append(neg_crop)

        ##### Crops for training the model 
        pos_crop = torch.FloatTensor([pos_crop]) # single positive crop so converting to list
        neg_crops = torch.FloatTensor(neg_crops_list) # torch.tensor(neg_crops_list)

        self.positive_dataset.pos_image_buffer.extend(pos_crop) # extend because single positive crop
        self.negative_dataset.neg_image_buffer.extend(neg_crops) # extend because multiple negative crops

        print(len(self.positive_dataset.pos_image_buffer), "Positive Image Buffer")
        print(len(self.negative_dataset.neg_image_buffer), "Negative Image Buffer")

        ###### Predicted Categories
        pos_category = pos_crop_properties.pred_category
        self.positive_categories = np.append(self.positive_categories, pos_category)
        
        neg_categories = neg_crops_properties.pred_category
        self.negative_categories = np.append(self.negative_categories, neg_categories, 0)

        ###### Predicted Confidences
        pos_confidence = pos_crop_properties.pred_confidence
        self.positive_confidences = np.append(self.positive_confidences, pos_confidence)
        
        neg_confidences = neg_crops_properties.pred_confidence
        self.negative_confidences = np.append(self.negative_confidences, neg_confidences, 0)

        if self.start == True:
            # To deal with empty data loader case throwing a pytorch error
            self.positive_dataset.pos_image_buffer.pop(0)
            self.negative_dataset.neg_image_buffer.pop(0)
            self.start = False
        
        self.positive_dataset.pos_image_buffer = np.array(self.positive_dataset.pos_image_buffer, dtype=object)
        self.negative_dataset.neg_image_buffer = np.array(self.negative_dataset.neg_image_buffer, dtype=object)

        self.positive_dataset.pos_image_buffer = self.positive_dataset.pos_image_buffer.tolist()
        self.negative_dataset.neg_image_buffer = self.negative_dataset.neg_image_buffer.tolist()

        if self.start == False:
            assert len(self.positive_dataset.pos_image_buffer) == len(self.positive_categories) == len(self.positive_confidences)
            assert len(self.negative_dataset.neg_image_buffer) == len(self.negative_categories) == len(self.negative_confidences)
            
        if len(self.positive_dataset.pos_image_buffer) >= self.pos_batch_size and len(self.negative_dataset.neg_image_buffer) >= self.neg_batch_size:
            self.train_contrastive_clustering()
    
    def train_contrastive_clustering(self):
        '''
        Train the Contrastive Clustering Model online, on the positive and negative crops
        '''
        print("TRAINING!!!!!!!")
        for epoch in range(self.start_epoch, self.num_epochs):

            loss_epoch = 0
            lr = self.optimizer.param_groups[0]["lr"]

            for step, ((x_p, x_i1, x_i2), (x_n, x_j1, x_j2)) in enumerate(zip(self.pos_train_loader, self.neg_train_loader)):
                self.optimizer.zero_grad()

                x_p = x_p.type(torch.float32) # original positive image
                x_i1 = x_i1.type(torch.float32) # first augmentation of positive
                x_i2 = x_i2.type(torch.float32) # second augmentation of positive

                x_n = x_n.type(torch.float32) # original negative images
                x_j1 = x_j1.type(torch.float32) # first augmentation of negative
                x_j2 = x_j2.type(torch.float32) # second augmentation of negative

                x = torch.cat((x_p, x_n)) # all original images
                x_i = torch.cat((x_i1, x_j1)) # all first augmentations (p1, n1, n1, n1, .....)
                x_j = torch.cat((x_i2, x_j2)) # all second augmentations (p2, n2, n2, n2, ......)

                x = x.cuda(non_blocking=True)
                x_i = x_i.cuda(non_blocking=True)
                x_j = x_j.cuda(non_blocking=True)

                # import pdb
                # pdb.set_trace()

                # Train the model
                h1, h_i, h_j, z_i, z_j, c_i, c_j = self.cc_model(x, x_i, x_j)

                human_embeddings = h1[:self.pos_batch_size]
                
                # Calculate the loss and backpropagate
                loss_instance = self.criterion_instance(z_i, z_j)
                loss_cluster = self.criterion_cluster(c_i, c_j)
                loss_one_class = self.criterion_one_class(human_embeddings)

                loss = loss_instance + loss_cluster + loss_one_class
                loss.backward()

                self.optimizer.step()
                # print(f"Step [{step}/{(len(self.pos_train_loader)+len(self.neg_train_loader))}]\t Loss: {loss.item()}")

                self.writer.add_scalar("Loss/train_epoch", loss.item(), self.global_step)
                self.global_step += 1

                loss_epoch += loss.item()
            print(f"Epoch [{epoch}/{self.num_epochs}]\t Loss: {loss_epoch / (len(self.pos_train_loader)+len(self.neg_train_loader))}\t lr: {round(lr, 5)}")

        self.global_epoch += self.num_epochs

        self.model_save_path = os.path.join(self.model_save_dir, self.target_subcat+"_"+str(self.global_epoch)+".pth")
        torch.save(self.cc_model.state_dict(), self.model_save_path)
        print("Weights saved at: ", self.model_save_path)
      
    def process_image(self, image):
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        image = np.asarray(image, np.float32)
        # mean subtraction from image
        image -= self.mean
        # transpose to get channels first
        image = image.transpose((2, 0, 1))
        return image.copy()
     
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
    
class PositiveDataSet(data.Dataset):
    def __init__(self, transforms):
        self.pos_image_buffer = []
        self.transforms = transforms
    
    def __len__(self):
        return(len(self.pos_image_buffer))
    
    def __getitem__(self, index):
        data = self.pos_image_buffer[index]
        data = data.type(torch.uint8)
        image_1 = self.transforms(data)
        image_2 = self.transforms(data)
        return data, image_1, image_2
class NegativeDataSet(data.Dataset):
    def __init__(self, transforms):
        self.neg_image_buffer = []
        self.transforms = transforms
    
    def __len__(self):
        return(len(self.neg_image_buffer))
    
    def __getitem__(self, index):
        data = self.neg_image_buffer[index]
        data = data.type(torch.uint8)
        image_1 = self.transforms(data)
        image_2 = self.transforms(data)
        return data, image_1, image_2

if __name__ == '__main__':
    try:
        node = TrainContrastiveSimilarity()
    except rospy.ROSInterruptException:
        pass
