import cv2
import numpy as np 
import csv
import pandas as pd
import os
import moviepy.editor as mpy


ROOT = os.path.join(os.path.expanduser("~"), "hand_classification_data")

ANN_COL_ID = 42

def process_labels(label_path):
    '''
    Extract annotations from the annotated CSV file 

    Parameters:
                label_path (string): Path where the annotated CSV file is stored for a given participant
    Returns:
                labels (np.array): Array of labels for each frame of the video extracted from the labeled CSV file
    '''
    annotations = pd.read_csv(label_path)
    labels = annotations.iloc[:, ANN_COL_ID]
    labels = np.array(labels)
    return labels

def process_preds(pred_path, counts_path, class_map):
    pred_gt_array = np.load(pred_path, allow_pickle=True)
    counts_array =  np.load(counts_path, allow_pickle=True)
    preds = pred_gt_array[:,1]
    gts = pred_gt_array[:,2]
    pred_list = []
    gt_list = []
    for p,g in zip(preds, gts):
        pred = class_map[p]
        pred_list.append(pred)
        gt = class_map[g]
        gt_list.append(gt)
    pred_counts = counts_array[0]
    gt_counts = counts_array[1]
    return pred_list, gt_list, pred_counts, gt_counts

def annotate_video_pred(video_path, pred_list, gt_list, pred_counts, gt_counts, out_image_dir):
    vid = mpy.VideoFileClip(video_path)
    for i, ((tstamp, frame), pred, gt, pc, gc) in enumerate(zip(vid.iter_frames(with_times=True), pred_list, gt_list, pred_counts, gt_counts)):
        print(i, pred, gt)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        text = "GT: " + gt + " PRED: " + pred + " GT_COUNT: " + str(gc) + " PRED_COUNT: " + str(pc)

        annotated_frame = cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 5, cv2.LINE_AA)
        out_filename = os.path.join(out_image_dir, str(i)+'.png')
        cv2.imwrite(out_filename, annotated_frame)
    
def annotate_video_gt(video_path, labels, out_video_path):
    '''
    Read every frame of the video along with every element in the labels array, and print the label over each frame of the video

    Parameters:
                video_path (string): Path to where the video is stored for each participant
                labels (np.array): Array of labels for each frame of the video extracted from the labeled CSV file
                out_video_path (string): Path to store the output video
    '''
    vid = mpy.VideoFileClip(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(out_video_path, fourcc, 1, (1280, 720))

    for i, ((tstamp, frame), label) in enumerate(zip(vid.iter_frames(with_times=True), labels)):
            print(i, label)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            annotated_frame = cv2.putText(frame, label, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 5, cv2.LINE_AA)
            video.write(annotated_frame)

def main():
    P_ID = ["P_3", "P_5", "P_1"]
    for p_id in P_ID:
        print(p_id)
        LABEL_FILENAME = p_id + '_annotated.csv'
        VIDEO_FILENAME = p_id + '.webm'
        if p_id == "P_3":
            VIDEO_FILENAME = p_id + ".mp4"
        OUTPUT_FILENAME = p_id + '_gt_viz.mp4'
        ANNOTATED_IMAGE_DIR = p_id + '_annotated_images'
        data_dir = os.path.join(ROOT,  p_id)
        label_path = os.path.join(data_dir, LABEL_FILENAME)
        video_path = os.path.join(data_dir, VIDEO_FILENAME)
        out_video_path = os.path.join(data_dir, OUTPUT_FILENAME)
        out_image_dir = os.path.join(data_dir, ANNOTATED_IMAGE_DIR)
        if not os.path.exists(out_image_dir):
            os.mkdir(out_image_dir)
        class_map = {0:"None", 1:"PICK", 2:"NOT_PICK"}
        gt = False

        pred_path = '/home/scazlab/catkin_ws/src/recycling_stretch/scripts/hand_tracker/classifier/utils/preds_20.npy'
        counts_path = '/home/scazlab/catkin_ws/src/recycling_stretch/scripts/hand_tracker/classifier/utils/pred_gt_counts.npy'

        preds, labels, pred_counts, gt_counts = process_preds(pred_path, counts_path, class_map)

        annotate_video_pred(video_path, preds, labels, pred_counts, gt_counts, out_image_dir)


        if gt:
            # Extract labels from the annotated CSV file (for ground truth)
            labels = process_labels(label_path)

            # Superimpose the labels on each frame of the video and save the video
            annotate_video_gt(video_path, labels, out_video_path)


if __name__ == "__main__":
    main()
