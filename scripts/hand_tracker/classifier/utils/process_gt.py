import numpy as np 
import csv
import json
import os
import pandas as pd
import cv2
import moviepy.editor as mpy
from matplotlib import pyplot as plt
from csv import reader


ROOT = os.path.join(os.path.expanduser("~"), "hand_classification_data")
LABEL_FILENAME = "P_1.csv"

def process_label_csv(label_path):
    '''
    Reads the output CSV file of VGG Video Annotator and converts it into a dictionary

            Parameters:
                    label_path (str): Path to the annotation file
            Returns:
                    ann_dict (dict): Dictionary storing processed annotations in the format
                                    {"P_ID":{
                                        "Seg_ID":{
                                            "start_time": float,
                                            "end_time": float,
                                            "category": {"Activity": cat}
                                                }
                                            }
                                    }
    '''

    # VGG Video Annotator has the following fields in the Annotation CSV file
    # metadata_id, file_list, temporal_segment_start, temporal_segment_end,  metadata
    # Important: To avoid dataframe errors, delete the first row of the raw Annotation CSV file 
    # Important: While exporting annotations, select "Export ONLY Temporal Segments as CSV"

    # reading csv file
    annotations = pd.read_csv(label_path)

    # get available annotated participant data
    filename_list = []
    for idx, row in annotations.iterrows():
        # convert string to list
        filename = json.loads(row[1])
        filename_list.append(filename[0])

    # select unique filenames to get participant ids
    ann_p_ids = list(set(filename_list))

    # iterate through rows of CSV file and convert to dict
    ann_dict = {}
    for p_id in ann_p_ids:
        seg_count = 0
        seg_dict = {}
        for idx, row in annotations.iterrows():
            filename = json.loads(row[1])
            sub_seg_dict = {}
            if filename[0] == p_id:
                sub_seg_dict["start_time"] = row[2]
                sub_seg_dict["end_time"] = row[3]
                sub_seg_dict["category"] = json.loads(row[4])
                seg_count = seg_count + 1
                seg_key = "Seg_" + str(seg_count)
                seg_dict[seg_key] = sub_seg_dict
        ann_key = p_id.split('.')[0]
        ann_dict[ann_key] = seg_dict   
    return ann_dict

def extract_video_timestamp(video_filename, p_id):
    '''
    Reads a participant's video and extacts the timestamp of each frame

            Parameters:
                    video_filename (str): Path to the video
                    p_id (str): Participant ID
            Returns:
                    timestamp_dict (dict): Dictionary storing the timestamps at which each frame is obtained in the format
                                            {"P_ID": [t_0, t_1, ......., t_num_frames]}
    '''
    vid = mpy.VideoFileClip(video_filename)
    timestamp_dict = {}
    timestamp_list = []

    # https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video
    for i, (tstamp, frame) in enumerate(vid.iter_frames(with_times=True)):
        timestamp_list.append(tstamp) # tstamp%60
    timestamp_dict[p_id] = timestamp_list
    return timestamp_dict

def read_hand_keypoints(kp_csv_filename, p_id):
    '''
    Reads a participant's hand tracking CSV file and converts it to a dictionary

            Parameters:
                    kp_csv_filename (str): Path to the keypoint CSV file
                    p_id (str): Participant ID
            Returns:
                    kp_dict (dict): Dictionary storing the keypoints at each frame in the format
                                            {"P_ID": [[x0, y0, x1, y1, ...., x20, y20], 
                                                     [x0, y0, x1, y1, ...., x20, y20], 
                                                     ....
                                                     ]}
    '''
    kp_dict = {}
    kp_list = []
    # reading csv file
    with open(kp_csv_filename, 'rt', encoding='utf-8') as f:
        csv_reader = reader(f)
        for line in csv_reader:
            kp_list.append(line)
    kp_dict[p_id] = kp_list
    return kp_dict



def align_keypoints_with_annotations(timestamp_dict, kp_dict, annotation_dict, out_csv_file):
    '''
    Index annotated timestamp segments into the keypoint dictionary and save the annotated CSV file

            Parameters:
                    timestamp_dict (dict): Dictionary mapping timestamps to their corresponding frames and participant IDs
                    kp_dict (dict): Dictionary mapping keypoints to participant IDs
                    annotation_dict (dict): Processed annotation dictionary
                    out_csv_file (str): Path to output CSV file
    '''

    
    #df = pandas.DataFrame([])
    final_list = list()
    temp_list = ['None']*42
    new_start = 0
    target_index = [len(v) for v in timestamp_dict.values()][0]
                    
    for sid, ann in annotation_dict.items():
        print("SEGMENT", sid)
        start_time = float(ann['start_time'])
        end_time = float(ann['end_time'])
        category = ann["category"]
        category = category["Activity"]
        default_category = "None"

        # iterate over timestamp and keypoint list to find corresponding labeled segments 
        # according to start times and end times
        for (k1, ts), (k2, kp) in zip(timestamp_dict.items(), kp_dict.items()):
            ts = np.array(ts)

            # calculate start_index and end_index by indexing into the time array using the start_time and end_time
            start_idx = np.where(ts >= start_time)[0][0]
            end_idx = np.where(ts <= end_time)[0][-1]
            print("*****", start_idx,end_idx)
            if start_idx > end_idx:
                print("CHECK DATA!!!!!!!!!!")

            # index keypoint list using calculated start index and end index
            pre_start_segment = kp[new_start:start_idx]
            segment = kp[start_idx:end_idx]
            new_start = end_idx
        
            a = [['None']*43 for _ in pre_start_segment]
            print('pre_start_segment', len(a), len(pre_start_segment))

            b = [inner_list + [category] for inner_list in segment if len(inner_list)!=0]
           
            c = [temp_list + [category] for inner_list in segment if len(inner_list)==0]
            print('segment', len(b), len(c), len(b)+len(c), len(segment))
 
            final_list.extend(a)
            final_list.extend(b)
            final_list.extend(c)
            print(len(final_list), start_idx, end_idx)

    if end_idx != target_index:
        d = [['None']*43 for _ in range(target_index-end_idx)]
        final_list.extend(d)

    print('total_length',len(final_list))

    df = pd.DataFrame(final_list)
    df.to_csv(out_csv_file, index=False)
 
def main():
    label_path = os.path.join(ROOT, "labels", LABEL_FILENAME)

    # Read the annotation CSV file and process it
    annotation_dict = process_label_csv(label_path)

    for p_id in annotation_dict.keys():
        data_dir = os.path.join(ROOT, p_id)
        video_filename = os.path.join(data_dir, p_id + '.webm')
        kp_csv_filename = os.path.join(data_dir, p_id + '.csv')

        # Extract timestamp for each frame in a given video
        timestamp_dict = extract_video_timestamp(video_filename, p_id)

        # Read hand tracking keypoints

        kp_dict = read_hand_keypoints(kp_csv_filename, p_id)


        print(len(kp_dict[p_id]), "Num keypoints")
        print(len(timestamp_dict[p_id]), "Num timestamps")

        # Sanity Check
        assert len(kp_dict[p_id]) == len(timestamp_dict[p_id])

        out_csv_file = os.path.join(data_dir, p_id + '_annotated.csv')
        # Align hand-tracking keypoints with annotations
        align_keypoints_with_annotations(timestamp_dict, kp_dict, annotation_dict[p_id], out_csv_file)


        print(len(kp_dict[p_id]), "Num keypoints")
        print(len(timestamp_dict[p_id]), "Num timestamps")




if __name__ == "__main__":
    main()