import numpy as np 
from hand_detector import HandDetector
from realsensecv import RealsenseCapture
from PIL import Image as im
import cv2
import time
import itertools
import csv
import os

ROOT = os.path.expanduser("~")
FPS = 30
 
def record_data(detector, participant_id, participant_data_path, realsense, rgb_frames, depth_frames):
    '''
    Opens camera stream, records video, detects hand keypoints and saves them

            Parameters:
                    detector (obj): Hand Detector class object
                    participant_id (str): Participant ID entered by the user
                    participant_data_path (str): Path to save participant data
                    realsense (bool): True if RealSense camera is available/ False for webcam testing

    '''
    master_kp_list = []

    # video_filename = "P_" + participant_id + '.webm'
    # video_path = os.path.join(participant_data_path, video_filename)
    # print(video_path)

    # if realsense:
    #     depth_video_filename = "depth_P_" + participant_id + '.webm'
    #     depth_video_path = os.path.join(participant_data_path, depth_video_filename)

    # if realsense:
    #     FRAME_WIDTH = 1280
    #     FRAME_HEIGHT = 720
    # else:
    #     FRAME_WIDTH = 640
    #     FRAME_HEIGHT = 480

    # # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'VP90')
    # out_rgb = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    # if realsense:
    #     out_depth = cv2.VideoWriter(depth_video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    pTime = 0
    cTime = 0
    total_time = 0

    if realsense:
        cap = RealsenseCapture()
        # Property setting
        # cap.WIDTH = FRAME_WIDTH
        # cap.HEIGHT = FRAME_HEIGHT
        cap.FPS = FPS
        # Unlike cv2.VideoCapture(), do not forget cap.start()
        cap.start()
        print("Cap started!")
    else:
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture('output.mp4')

    while True: # cap.is    Opened():
        success, frame = cap.read()

        if not realsense:
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if realsense:
            img = frame[0]
            img_depth = frame[1]
             # in the heat map conversion
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.08), cv2.COLORMAP_JET)
        else:
            img = frame


        # out_rgb.write(img)
        rgb_frames.append(img)
        # depth_frames.append(img_depth.astype(np.uint8))
        depth_frames.append(depth_colormap)
        # print(videos)
        
        if realsense:
            pass
            # out_depth.write(depth_colormap)
        #print("Writing"f)

        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        
        # if len(lmlist) != 0: 
        # Post-Process landmarks to save them in appropriate format          
        kp_list = postprocess_landmarks(lmlist)

        # Aggregate processed landmarks from all frames into a single list of lists
        master_kp_list.append(kp_list)
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if success:
            cv2.imshow("Image", img)

        key = cv2.waitKey(1) 

        # Press q to quit the video capture and save the output   
        if key == ord('q'):
            break
        
    # Store output keypoints in CSV file
    store_csv_file(master_kp_list, participant_id, participant_data_path)  

    # Close the window / Release webcam
    cap.release()
    
    # After we release our webcam, we also release the output
    # out_rgb.release() 
    # if realsense:
    #     out_depth.release()
    
    # De-allocate any associated memory usage 
    cv2.destroyAllWindows() 

    return 
        
def postprocess_landmarks(hand_landmarks): 
    '''
    Post-process raw landmarks in appropriate format

            Parameters:
                    hand_landmarks (list): 21 lists of x,y positions of each landmark 
                                            eg: [0, x0, y0], [1, x1, y1], ........[20, x20, y20]
            Returns:
                    xy_list (list): list of x,y positions of all keypoints in the fframe
                                    eg: [x0, y0, x1, y1, .........., x20, y20]
    '''
    tuple_list = []
    for idx, kp in enumerate(hand_landmarks):
        x = kp[1]
        y = kp[2]
        xy_tuple = (x,y)
        tuple_list.append(xy_tuple) # [(x0, y0), (x1, y1), ....]
    xy_list = list(itertools.chain(*tuple_list)) # [x0, y0, x1, y1, ....]
    return xy_list

def store_csv_file(kp_list, participant_id, participant_data_path):
    '''
    Save all keypoints from a video into CSV format

            Parameters:
                    kp_list (list(list)): list of all keypoints aggregated for all frames in a video
                                        eg: [[x0, y0, x1, y1, .........., x20, y20], 
                                            [x0, y0, x1, y1, .........., x20, y20], 
                                            [x0, y0, x1, y1, .........., x20, y20], 
                                            ......]
                    participant_id (str): Participant ID entered by the user
                    participant_data_path (str): Path to save participant data
    '''
    csv_filename = "P_" + participant_id + '.csv'
    csv_path = os.path.join(participant_data_path, csv_filename)
    with open(csv_path, 'w') as f:
        write = csv.writer(f)
        write.writerows(kp_list)
        print("Done Writing CSV!")        

def write_video(rgb_frames, depth_frames, participant_id, participant_data_path):
    data = im.fromarray(rgb_frames[10])     
    data.save('gfg_dummy_pic.png')

    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    channel = 3
    video_filename = "P_" + participant_id + '.webm'
    video_path = os.path.join(participant_data_path, video_filename)

    depth_video_filename = "depth_P_" + participant_id + '.webm'
    depth_video_path = os.path.join(participant_data_path, depth_video_filename)

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out_rgb = cv2.VideoWriter(video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    out_depth = cv2.VideoWriter(depth_video_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    print("LENGTH OF FRAMES {}".format(len(rgb_frames)))
    for i, frame in enumerate(rgb_frames):        
        out_rgb.write(frame)     
    print("DONE WRITTING RGB VIDEO!") 

    print("LENGTH OF Depth FRAMES {}".format(len(depth_frames)))
    for i, frame in enumerate(depth_frames):        
        out_depth.write(frame)     
    print("DONE WRITTING DEPTH VIDEO!")  

def main():
    rgb_frames = [] 
    depth_frames = [] 
    realsense = True
    detector = HandDetector()
    participant_id = input("Enter participant ID: ")
    participant_data_path = os.path.join(ROOT, "hand_classification_data", "P_"+participant_id)
    if not os.path.exists(participant_data_path):
        os.makedirs(participant_data_path)
    record_data(detector, participant_id, participant_data_path, realsense, rgb_frames, depth_frames)
    write_video(rgb_frames, depth_frames, participant_id, participant_data_path)
    


if __name__ == "__main__":
    main()