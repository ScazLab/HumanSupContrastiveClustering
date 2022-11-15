from cv2 import data
import numpy as np
import cv2
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os


ROOT = os.path.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
P_ID = "P_25"
LABEL_FILENAME = P_ID + '_annotated.csv'
VIDEO_FILENAME = P_ID + '.webm'

ANN_COL_ID = 42

def read_lm_csv(label_path):
    data = pd.read_csv(label_path)
    lm = data.iloc[:, 0:ANN_COL_ID]
    lm = np.array(lm)
    return lm

def draw_box_over_keypoints(lm_list, image_dir, box_viz_dir):
    for id, lm in enumerate(lm_list):
        image_id = P_ID+'_frame_'+str(id)+'.png'
        image_file = os.path.join(image_dir, image_id)
        img = cv2.imread(image_file)
        
        h, w, c = img.shape
        coord_list = []
        for i in range(0, len(lm), 2):
            coord = lm[i:i+2]
            x = coord[0]
            y = coord[1]
            if x!='None' and y!='None':
                coord_list.append([int(x), int(y)])
                cx, cy = int(x), int(y) 
                img = cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        coord_list = np.array(coord_list)
        print(id)
        if len(coord_list) > 0:
            rect = cv2.minAreaRect(coord_list)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            result = img.copy()
            cv2.drawContours(result, [box], 0, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(box_viz_dir, str(id)+'.png'), result)
        
            
        # cv2.imwrite('keypoints.png', img)


def main():
    data_dir = os.path.join(ROOT,  P_ID)
    label_path = os.path.join(data_dir, LABEL_FILENAME)
    image_dir = os.path.join(data_dir, 'Images')
    box_viz_dir = os.path.join(data_dir, 'Box_Viz')
    if not os.path.exists(box_viz_dir):
        os.mkdir(box_viz_dir)
    lm = read_lm_csv(label_path)
    draw_box_over_keypoints(lm, image_dir, box_viz_dir)


if __name__ == "__main__":
    main()