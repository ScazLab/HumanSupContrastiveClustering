#!/usr/bin/env python2

# Message imports go here
from email import header
from recycling_stretch.msg import DetectedBox, DebugRealPose, ArucoRefMarker, HitList, CalibrationParams, SimDetectedBox, HitSelection
from recycling_stretch.srv import PushObject, PushObjectRequest, PushObjectResponse, \
                                PullObject, PullObjectRequest, PullObjectResponse, \
                                LiftArm, LiftArmRequest, LiftArmResponse
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32, Int16
from tf2_geometry_msgs import PoseStamped
from geometry_msgs.msg import Pose
# Service imports go here

# All other imports go here
from tf.transformations import quaternion_from_euler
from operator import attrgetter
import math
import ros_numpy
import sys
import copy
import pdb
import rospy
import cv2
import numpy as np
import message_filters
from shapely.geometry import Polygon, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros

# Hyper-parameters go here
SLEEP_RATE = 10

MARKER_PUSH_PLATE_OFFSET = 0.0889
PIXEL_TO_REAL_CONVERSION_RATIO = 0.00204494274
HIT_PLATE_WIDTH = 79 * PIXEL_TO_REAL_CONVERSION_RATIO
HIT_PLATE_HEIGHT = 1.0 # adjust to cover entire length of conveyor belt

# positive if too early, negative if late, adjust accordingly
PUSH_ADJUSTMENT_PARAMETER = 2.0
PULL_ADJUSTMENT_PARAMETER = 1.5

ROBOT_PUSH_POLYGON = Polygon()
ROBOT_PULL_POLYGON = Polygon()

PUSH_STATE, PULL_STATE, SELECTION_STATE, AVOID_OBJECT = [1, 2, 3, 4]
OBJECT_REMOVAL_TIME_UPPER_BOUND = 8 # 10 seconds

class Recyclable:
    def __init__(self, pose, height, width, angle_of_rotation, min_depth, similarity_score, pred_recyclable_category, header_time_stamp, hit_polygon_pixel=None):
        self.pose = pose 
        self.center_position = pose.position
        self.height = height
        self.width = width
        self.angle_of_rotation = angle_of_rotation
        self.min_depth = min_depth  
        self.similarity_score = similarity_score
        self.pred_recyclable_category = pred_recyclable_category
        self.header_time_stamp = header_time_stamp

        self._hit_score = 0.0
        self._hit_time = 0.0
        self._hit_casualties = []
        self._hit_polygon_pixel = hit_polygon_pixel
        self._hit_polygon_real = None

        self._push_score = 0.0
        self._pull_score = 0.0

        self.request = "NONE"
    def calculate_scores(self, belt_velocity, marker_position, dominant_positive_category):
        # calculate score for pushing
        self._push_score = self.update_score(belt_velocity, ROBOT_PUSH_POLYGON, similarity_weight=4.0, pred_recyclable_category_weight=1.0, hit_casualty_overlap_weight=0.1)

        # calculate score for pulling
        self._pull_score = self.update_score(belt_velocity, ROBOT_PULL_POLYGON, similarity_weight=4.0, pred_recyclable_category_weight=1.0, hit_casualty_overlap_weight=0.1)
        
        if (marker_position.y + MARKER_PUSH_PLATE_OFFSET) - self.center_position.y < 0:
            self._pull_score *= -1
        else:
            self._push_score *= -1

        self._max_score = max(self._push_score, self._pull_score)
        if self._max_score == self._push_score:
            self.request = PUSH_STATE
        else:
            self.request = PULL_STATE

    def update_score(self, belt_velocity, robot_hit_polygon, similarity_weight=None, pred_recyclable_category_weight=None, hit_casualty_overlap_weight=None):

        similarity_score_sum = 0.0
        pred_recyclable_category_sum = 0.0
        hit_casualty_overlap_sum = 0.0

        for item in self._hit_casualties:
            casualty_polygon = item.construct_hit_polygon_adjusted(belt_velocity, self._hit_time)
            casualty_overlap = compute_overlap(robot_hit_polygon, casualty_polygon) / casualty_polygon.area
            if casualty_overlap > 0:
                hit_casualty_overlap_sum += casualty_overlap
                similarity_score_sum += item.similarity_score
                pred_recyclable_category_sum += item.pred_recyclable_category

        return ((similarity_score_sum * 0.5 + self.similarity_score) * similarity_weight) \
                + ((pred_recyclable_category_sum + self.pred_recyclable_category) * pred_recyclable_category_weight) \
                     + (hit_casualty_overlap_sum * hit_casualty_overlap_weight)

    def update_hit_time(self, belt_velocity, marker_position):
        # print(marker_position.x - HIT_PLATE_WIDTH)
        self._hit_time = abs(marker_position.x - HIT_PLATE_WIDTH - self.center_position.x) / belt_velocity

    def construct_hit_polygon_adjusted(self, belt_velocity, hit_time):
        # helper function to adjust bounding box to anticipated location after self._hit_time
        hit_polygon_real_adjusted = []

        # shift coordinates in the direction of the belt motion (x-axis)
        for point in self._hit_polygon_real.exterior.coords[:-1]:
            x, y = point
            x = x + hit_time * belt_velocity
            hit_polygon_real_adjusted.append((x, y))

        return Polygon(hit_polygon_real_adjusted)

class RecyclablePlanner(object):
    def __init__(self):
        # Other things besides pubs, subs, and services
        # print("Waiting for belt velocity")
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(720.0))
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.state = AVOID_OBJECT
        self.prev_hit_call_time = None
        self.action_started = False
        self.original_time = 0
        print("WAITING FOR PUSH OBJECT SRV")
        rospy.wait_for_service("/recycling_stretch/push_object_srv")
        self.push_object_sp = rospy.ServiceProxy("/recycling_stretch/push_object_srv", PushObject)
        print("WAITING FOR PULL OBJECT SRV")
        rospy.wait_for_service("/recycling_stretch/pull_object_srv")
        self.pull_object_sp = rospy.ServiceProxy("/recycling_stretch/pull_object_srv", PullObject)

        print("WAITING FOR LIFT ARM SRV")
        rospy.wait_for_service("/recycling_stretch/lift_arm_srv")
        self.lift_arm_sp = rospy.ServiceProxy("/recycling_stretch/lift_arm_srv", LiftArm)

        self.arm_state = SELECTION_STATE
        self.lift_arm_default_state_position(AVOID_OBJECT)

        print("WAITING FOR BELT VELOCITY")
        self.belt_velocity = rospy.wait_for_message('/recycling_stretch/belt_velocity',
                                                       Float32)  # TODO change this to ros params
        
        self.belt_velocity = self.belt_velocity.data
        print("Belt velocity retrieved: {} m/s of type {}".format(self.belt_velocity, type(self.belt_velocity)))

        print("Waiting for lift calibration")
        self.lift_calibration_params = rospy.wait_for_message('/recycling_stretch/lift_calibration',
                                                                CalibrationParams)  # TODO change this to ros params

        print("Waiting for wrist calibation")
        self.wrist_calibration_params = rospy.wait_for_message('/recycling_stretch/wrist_calibration',
                                                                  CalibrationParams)  # TODO change this to ros params

        print("Waiting for dominant_positive_category")

        self.dominant_positive_category = rospy.wait_for_message('/recycling_stretch/dominant_human_cat', Int16)
        self.dominant_positive_category = self.dominant_positive_category.data
        print("All our wait for messages have arrived")
        
        self.wrist_extension_velocity = self.wrist_calibration_params.slope
        self.wrist_extension_intercept = self.wrist_calibration_params.intercept

        self.lift_velocity = self.lift_calibration_params.slope
        self.lift_intercept = self.lift_calibration_params.intercept

        self.wrist_marker_pose = Pose()
        self.recyclables = [] 

        self.rgb_image = None
        self.hit_timer = None
        # Publishers
        self.push_hit_polygon_visualization_pub = rospy.Publisher('/recycling_stretch/push_hit_polygon_marker_array', MarkerArray, queue_size=10)
        self.pull_hit_polygon_visualization_pub = rospy.Publisher('/recycling_stretch/pull_hit_polygon_marker_array', MarkerArray, queue_size=10)
        self.object_polygon_visualization_pub = rospy.Publisher('/recycling_stretch/object_polygon_marker_array', MarkerArray, queue_size=10)
        self.hit_selection_visualizer_pub = rospy.Publisher('/recycling_stretch/hit_selection_visualizer', Image, queue_size=10)
        # Subscribers
        self.belt_velocity_sub = rospy.Subscriber('/recycling_stretch/belt_velocity', Float32, callback=self.update_velocity_cb)
        self.rgb_img_sub = rospy.Subscriber('/camera2/color/image_raw', Image, callback=self.get_image_cb)
        self.target_marker_sub = rospy.Subscriber('/recycling_stretch/marker_position', ArucoRefMarker, callback=self.target_marker_cb)
        self.real_pose_sub = rospy.Subscriber('/recycling_stretch/real_pose', DebugRealPose, callback=self.get_object_list_cb)
        
        rospy.spin()
        
    def update_velocity_cb(self, belt_velocity):
        """
        Callback function to update 

        Parameters:
                        belt_velocity:
        """
        self.belt_velocity = belt_velocity.data

    def reset_object_selection_cb(self):
        self.state = AVOID_OBJECT

        print("SWITCHING STATES TO {}".format(self.state))

    def get_image_cb(self, ros_rgb_image):
        # print("TIME OF CURRENT MESSAGE (ROSTIME) {}".format(ros_rgb_image.header.stamp))
        # print("DIFFERENCE OF CURRENT AND ORIGINAL MESSAGE {}".format(abs(self.original_time - ros_rgb_image.header.stamp)))
        self.rgb_image = ros_numpy.numpify(ros_rgb_image)

        # OpenCV expects bgr images, but numpify by default returns rgb images.
        self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)

        current_time = rospy.Time.now().to_sec()
        if self.prev_hit_call_time is not None:
            print("TIME REMAINING TO HIT {} secs".format(self.prev_hit_call_time - current_time))
            
            # drop arm to prepare for hit
            if self.prev_hit_call_time - 2 > current_time - 0.1 and self.prev_hit_call_time - 2 < current_time + 0.5 and not self.action_started:
                self.lift_arm_default_state_position(SELECTION_STATE)

            # send hit request
            elif self.prev_hit_call_time > current_time - 0.1 and self.prev_hit_call_time < current_time + 0.3 and not self.action_started:
                if self.arm_state == AVOID_OBJECT:
                    self.lift_arm_default_state_position(SELECTION_STATE)
                if self.state == PUSH_STATE:
                    self.remove_object(0.51, 0.0, PUSH_STATE)
                elif self.state == PULL_STATE:
                    self.remove_object(0.05, -0.03, PULL_STATE)
                print("SENDING REQUEST TO ROBOT AT {}".format(current_time))
                self.action_started = True
                self.lift_arm_default_state_position(AVOID_OBJECT)

            # return arm to avoid object state
            elif current_time - self.prev_hit_call_time > OBJECT_REMOVAL_TIME_UPPER_BOUND:
                print("OBJECT REMOVAL IS COMPLETE AT {}".format(current_time))
                self.reset_object_selection_cb()
                self.prev_hit_call_time = None
                self.action_started = False
    
    def target_marker_cb(self, msg):
        """
        Construct push and pull robot polygon
        """
        global ROBOT_PULL_POLYGON
        global ROBOT_PUSH_POLYGON
        # angle_rotation = math.atan2(self.wrist_extension_velocity, self.belt_velocity)
        angle_rotation = math.atan2(0.8, 0.3)
        if msg.label == "top_wrist_marker":
            self.wrist_marker_pose = msg.pose
            bbox_bottom_left = (msg.pose.position.x + HIT_PLATE_WIDTH / 2, msg.pose.position.x)
            bbox_bottom_right = (msg.pose.position.y - HIT_PLATE_WIDTH / 2, msg.pose.position.y)

            bbox_top_left_push = (msg.pose.position.x - HIT_PLATE_HEIGHT * math.cos(angle_rotation),
                                  msg.pose.position.y + HIT_PLATE_HEIGHT * math.sin(angle_rotation))
            bbox_top_right_push = (bbox_top_left_push[0] - HIT_PLATE_WIDTH, bbox_top_left_push[1])

            bbox_top_left_pull = (msg.pose.position.x - HIT_PLATE_HEIGHT * math.cos(angle_rotation),
                                  msg.pose.position.y - HIT_PLATE_HEIGHT * math.sin(angle_rotation))
            bbox_top_right_pull = (bbox_top_left_pull[0] - HIT_PLATE_WIDTH, bbox_top_left_pull[1])

            ROBOT_PUSH_POLYGON = Polygon([bbox_bottom_left, bbox_bottom_right, bbox_top_right_push, bbox_top_left_push, bbox_bottom_left])
            self.push_hit_polygon_visualization_pub.publish([self.get_ros_marker(ROBOT_PUSH_POLYGON, msg.pose, "camera1_color_optical_frame")])

            ROBOT_PULL_POLYGON = Polygon([bbox_bottom_left, bbox_bottom_right, bbox_top_right_pull, bbox_top_left_pull, bbox_bottom_left])
            self.pull_hit_polygon_visualization_pub.publish([self.get_ros_marker(ROBOT_PULL_POLYGON, msg.pose, "camera1_color_optical_frame")])

    def get_object_list_cb(self, msg):
        """
        Check whether the enough time has elapsed to start planning for the next hit
        """
        print("TIME OF ORIGINAL MESSAGE (ROSTIME) {}".format(msg.header.stamp))
        self.original_time = msg.header.stamp

        self.recyclables = []
        if len(msg.sim_detected_box_data.rbboxes) == len(msg.poses):
            for rbboxes, pose, similarity, confidence, pred_category in zip(msg.sim_detected_box_data.rbboxes, msg.poses, msg.sim_detected_box_data.similarity, msg.sim_detected_box_data.scores, msg.sim_detected_box_data.pred_category):
                # transform pixel rbboxes to real_space
                top_left = Point(rbboxes.top_left_x, rbboxes.top_left_y)
                top_right = Point(rbboxes.top_right_x, rbboxes.top_right_y)
                bottom_left = Point(rbboxes.bottom_left_x, rbboxes.bottom_left_y)
                bottom_right = Point(rbboxes.bottom_right_x, rbboxes.bottom_right_y)
                coords = [top_left, top_right, bottom_left, bottom_right, top_left]
                listarray = []

                for pp in coords:
                    listarray.append((int(pp.x), int(pp.y)))

                (center_x, center_y), (width, height), angle_of_rotation = cv2.minAreaRect(np.array(listarray))

                width = width * PIXEL_TO_REAL_CONVERSION_RATIO
                height = height * PIXEL_TO_REAL_CONVERSION_RATIO

                angle_of_rotation = math.radians(angle_of_rotation)
                
                # fill in all real data
                item = Recyclable(pose.pose, width, height, angle_of_rotation, pose.pose.position.z, similarity_score=similarity, pred_recyclable_category=pred_category, header_time_stamp=msg.header)
                item._hit_polygon_pixel = listarray[:-1]
                self.recyclables.append(item)
        else:
            print(len(msg.sim_detected_box_data.rbboxes))
            print(len(msg.poses))
            print(msg.poses)
            print(msg.sim_detected_box_data.rbboxes)
            sys.exit(0)
            raise Exception

        object_marker_array = MarkerArray()
        for i, item in enumerate(self.recyclables):
            print("Amount of Items: {}".format(len(self.recyclables)))
            object_marker_ps = PoseStamped()
            object_marker_ps.pose = item.pose
            object_marker_ps.header = msg.header
            # pdb.set_trace()
            pose_in_cam1 = self.tf_buffer.transform(object_marker_ps, "camera1_color_optical_frame", rospy.Duration(1))
            item.center_position = pose_in_cam1.pose.position

            exterior_coords = cv2.boxPoints(((item.center_position.x, item.center_position.y), (width, height), angle_of_rotation))
            item._hit_polygon_real = Polygon(exterior_coords)

            distance_from_gripper = math.sqrt((pose_in_cam1.pose.position.x - self.wrist_marker_pose.position.x) ** 2 + \
                (pose_in_cam1.pose.position.y - self.wrist_marker_pose.position.y) ** 2 + \
                    (pose_in_cam1.pose.position.z - self.wrist_marker_pose.position.z) **2)

            # print("Center: {} Height: {} Width: {}".format(item.center_position, item.height, item.width))

            object_marker_array.markers.append(self.get_object_ros_marker(item, item._hit_polygon_real.centroid, "camera1_color_optical_frame", i))

        # self.object_polygon_visualization_pub.publish(object_marker_array)
        if self.state == AVOID_OBJECT:
            print("SELECTING OBJECT...")
            self.select_object(msg)
        

    def select_object(self, msg):
        # print(len(self.recyclables))
        """
        Planning Algorithm given world state
        """
        object_marker_array = MarkerArray()
        none_dominant = True
        hit_list=[]
        # for each item, loop through all items in self.recyclables, construct anticipated position, compute overlap with hit_plate, add to casualties
        for i, item in enumerate(self.recyclables):
            item.update_hit_time(self.belt_velocity, self.wrist_marker_pose.position)  

            if int(item.pred_recyclable_category) == self.dominant_positive_category and item.similarity_score > 0.4:
                print("FOUND OBJECT OF DOMINANT POSITIVE CATEGORY")
                hit_list.append(item)
                none_dominant = False
            
            for j, casualty in enumerate(self.recyclables):
                if i == j:
                    continue
                else: 
                    casualty_object = casualty.construct_hit_polygon_adjusted(self.belt_velocity, item._hit_time)

                    casualty_overlap_push = compute_overlap(casualty_object, ROBOT_PUSH_POLYGON) / casualty_object.area
                    casualty_overlap_pull = compute_overlap(casualty_object, ROBOT_PULL_POLYGON) / casualty_object.area

                    if casualty_overlap_pull or casualty_overlap_push != 0.0:
                        item._hit_casualties.append(casualty)    
        if none_dominant:
            return
        if len(self.recyclables) > 0:
            for i, item in enumerate(hit_list):
                item.calculate_scores(self.belt_velocity, self.wrist_marker_pose.position, self.dominant_positive_category)
                print("ITEM HAS PUSH SCORE: {} PULL SCORE: {}".format(item._push_score, item._pull_score))

            hit_object = max(hit_list, key=attrgetter('_max_score'))
            
            print("MESSAGE TIME AT {}".format(hit_object.header_time_stamp))
            print("ROSPY TIME RIGHT NOW {}".format(rospy.Time.now().to_sec()))
            
            self.hit_selection_visualizer_pub.publish(self.cv2_to_imgmsg(self.draw_rboxes(self.rgb_image, hit_object, hit_object._max_score, hit_object.similarity_score, (0, 255, 0), 1)))
            
            if hit_object.request == PUSH_STATE:
                distance_wrist_object = abs(self.wrist_marker_pose.position.y - hit_object.center_position.y)
                hit_delay = self.wrist_extension_velocity * distance_wrist_object + self.wrist_extension_intercept

                adjusted_time_to_hit = hit_object._hit_time - hit_delay + PUSH_ADJUSTMENT_PARAMETER
                
                self.prev_hit_call_time = hit_object.header_time_stamp.stamp.secs + adjusted_time_to_hit

                self.state = PUSH_STATE

                print("SWITCHING STATES TO {}".format(self.state))
                print("HIT TIME: {}",format(hit_object._hit_time))
                print("HIT DELAY: {}",format(hit_delay))

            elif hit_object.request == PULL_STATE:
                distance_wrist_object_z = abs(self.wrist_marker_pose.position.z - hit_object.min_depth)
                print("DIFFERENCE IN DEPTH IS {}".format(distance_wrist_object_z))
                distance_wrist_object_y = abs(self.wrist_marker_pose.position.y - hit_object.center_position.y)
                
                hit_delay_wrist = self.wrist_extension_velocity * distance_wrist_object_y + self.wrist_extension_intercept
                hit_delay_lift = self.lift_velocity * distance_wrist_object_z + self.lift_intercept
                if hit_delay_lift < 0:
                    hit_delay_lift = 0.0
                adjusted_time_to_hit = hit_object._hit_time - hit_delay_wrist + PULL_ADJUSTMENT_PARAMETER 
                
                self.prev_hit_call_time = hit_object.header_time_stamp.stamp.secs + adjusted_time_to_hit

                self.state = PULL_STATE

    def remove_object(self, wrist_extension, lift_extension, request):
        """
        Function to send service call corresponding to the request number

        Parameters:
                        wrist_extension (float): value (in meters) to extend telescopic arm
                        lift_extension (float): value (in meters) to lift telescopic arm
                        request (int): state value for type of request (push = 1, pull = 2)
        """
        print("REMOVING OBJECT")
        if request == PUSH_STATE:
            self.push_object_sp(wrist_extension=wrist_extension)
        elif request == PULL_STATE:
            self.pull_object_sp(wrist_extension=wrist_extension, lift_extension=lift_extension)

    def lift_arm_default_state_position(self, request):
        '''
        Function to bring robot arm back to particular state right before and after the removal to ensure limited stream intervention

        Parameters:
                    lift_extension float value for lift extension of the telescopic arm (positive = up, negative = down)
        '''
        if request == AVOID_OBJECT and self.arm_state == SELECTION_STATE:
            self.lift_arm_sp(lift_extension=0.075)
            self.arm_state = AVOID_OBJECT
        elif request == SELECTION_STATE and self.arm_state == AVOID_OBJECT:
            self.lift_arm_sp(lift_extension=-0.075)
            self.arm_state = SELECTION_STATE

    def get_ros_marker(self, hit_polygon, pose, camera_frame):
        '''
        Helper function to vizualize the real-world position of the object in RVIZ

        Parameters:
                    line_list (list of Point messages)
        Return:
                    marker (message): Marker message representing the vizualization of the object in RVIZ
        '''

        self.marker = Marker()
        self.marker.type = self.marker.LINE_STRIP
        self.marker.action = self.marker.ADD
        self.marker.lifetime = rospy.Duration(0.2)
        self.marker.text = "robot hit_polygon"
        self.marker.header.frame_id = camera_frame
        self.marker.header.stamp = rospy.Time.now()

        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0
        self.marker.color.a = 0.33
        self.marker.scale.x = 0.02

        for i in range(5):
            x, y = hit_polygon.exterior.coords[i]
            start_point = Point(x, y, pose.position.z)
            self.marker.points.append(start_point)
        # x, y = hit_polygon.exterior.coords[0]
        # start_point = Point(x, y, pose.position.z)
        # self.marker.points.append(start_point)

        
        return self.marker

    def get_object_ros_marker(self, item, center_pose, camera_frame, object_id):
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
        self.marker.text = "object rbbox"
        self.marker.header.frame_id = camera_frame
        self.marker.header.stamp = rospy.Time.now()
        self.marker.id = object_id

        self.marker.scale.x = item.height
        self.marker.scale.y = item.width
        self.marker.scale.z = 0.005 # half a centimeter tall

        self.marker.color.r = 0
        self.marker.color.g = 255
        self.marker.color.b = 0
        self.marker.color.a = 0.33

        self.marker.pose.position.x = center_pose.x
        self.marker.pose.position.y = center_pose.y
        self.marker.pose.position.z = 0.5

        # Euler in RPY
        q = quaternion_from_euler(0, 0, item.angle_of_rotation)
        

        self.marker.pose.orientation.x = q[0]       
        self.marker.pose.orientation.y = q[1]       
        self.marker.pose.orientation.z = q[2]       
        self.marker.pose.orientation.w = q[3]       
        

        return self.marker

    def draw_rboxes(self, image, hit_polygon, hit_score, similarity_score, color, thickness, put_text=True, put_score=True):
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
            for item in self.recyclables:
                display_polygon_array = np.array(item._hit_polygon_pixel, np.int32)
                cv2.polylines(image,[display_polygon_array], True, color, thickness)

                if put_text: 
                    cv2.putText(image, str(item.similarity_score), item._hit_polygon_pixel[0], cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
                if put_score:
                    cv2.putText(image, str(hit_score), item._hit_polygon_pixel[2], cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

            # cv2.circle(image, hit_polygon._hit_polygon_pixel[0], 1, (0, 255, 0), 20)
            if hit_polygon.request == PUSH_STATE:
                cv2.putText(image, "PUSH", (int(Polygon(hit_polygon._hit_polygon_pixel).centroid.x), int(Polygon(hit_polygon._hit_polygon_pixel).centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            else:
                cv2.putText(image, "PULL", (int(Polygon(hit_polygon._hit_polygon_pixel).centroid.x), int(Polygon(hit_polygon._hit_polygon_pixel).centroid.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

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

def compute_overlap(box1, box2):
    '''
    Helper function to calculate the overlap between two polygons

    Parameters:
                box1 (Polygon): Bounding box 1 converted to Polygon
                box2 (Polygon): Bounding box 2 converted to Polygon
    Returns:
           clear
                overlap (float): Overlap between the bounding boxes
    '''
    try:
        intersect_area = float(box1.intersection(box2).area)
        if intersect_area == 0.0:
            return 0.0
        overlap = intersect_area / (box1.area + box2.area - intersect_area)
        return overlap
    except Exception as e:
        return 0.0

def rotate_point(center_position, height, width, angle_of_rotation):
    '''
    Helper function to rotate bounding box with (x, y) coordinate by some angle of rotation

    Parameters:
                center_position (x, y): Coordinate of center of bounding box
                height (float): height of the bounding box
                width (float): width of the bounding box
                angle_of_rotation (float): theta for angle of rotation
    Returns:
                (x, y) (float, float): Coordinate of new center of rotated bounding box
    '''
    x = width
    y = height

    rotatedX = x * math.cos(angle_of_rotation) - y * math.sin(angle_of_rotation)
    rotatedY = x * math.sin(angle_of_rotation) + y * math.cos(angle_of_rotation)

    x = rotatedX + center_position.x
    y = rotatedY + center_position.y

    return x, y

def main():
    rospy.init_node("object_selection_planner")

    hit_planner_node = RecyclablePlanner()
    rate = rospy.Rate(SLEEP_RATE)
    # This while loop is what keeps the node from dieing
    while not rospy.is_shutdown():
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
