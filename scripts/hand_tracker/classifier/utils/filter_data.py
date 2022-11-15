import numpy as np
from numpy.lib.type_check import imag 
import pandas as pd
import csv
import os
import os.path as osp

def eliminate_confusing_instances(labels_list, image_ids):
    indexes = []
    remove_index = []
    eliminate = 7
    next_pick = False
    for i, label in enumerate(labels_list):
        if label == 'PICK':
            if len(indexes) == 0:
                continue

            # eliminating 3 succeeding NOT_PICK/None after previous PICK sequence
            if next_pick:
                remove_index.append(indexes[:eliminate])
            next_pick = True  
             
            #  eliminating 3 preceeding NOT_PICK/None before current PICK sequence
            remove_index.append(indexes[-eliminate:])
            indexes = []

        else:
            indexes.append(i)
    print(image_ids)
    new_image_ids = np.delete(image_ids, remove_index)
    labels_array = np.array(labels_list)
    filtered_labels_array = np.delete(labels_array, remove_index)
    return new_image_ids, filtered_labels_array

def create_balanced_dataset(filtered_labels_array, new_image_ids):
        
    pick_array = filtered_labels_array[filtered_labels_array=='PICK']
    pick_idx = np.where(filtered_labels_array=='PICK')
    not_pick_array = filtered_labels_array[filtered_labels_array=='NOT_PICK']
    none_array = filtered_labels_array[filtered_labels_array=='None']

    np_idx = np.random.randint(0, len(not_pick_array), size=len(pick_array))
    none_idx = np.random.randint(0, len(none_array), size=len(pick_array))
    all_idx = np.hstack((np_idx, none_idx, pick_idx[0]))

    new_image_ids = new_image_ids[all_idx]
    filtered_labels_array = filtered_labels_array[all_idx]
    return new_image_ids

def main():
    root = osp.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
    list_path = os.path.join(root, 'all_eval_file_list.text')
    boundary_list_path = os.path.join(root, 'boundary_removed_eval_file_list.text')
    balanced_list_path = os.path.join(root, 'balanced_eval_file_list.text')
    # p_ids = ['P_7', 'P_20', 'P_22', 'P_23', 'P_24', 'P_25', 'P_27', 'P_42', 'P_74', 'P_77']
    p_ids = ['P_1', 'P_3', 'P_5']
    image_ids = [i_id for i_id in open(list_path)]
    image_ids = np.array(image_ids)
    ann_col_id = 42
    labels_list = []

    for p in p_ids:
        print(p)
        label_file = osp.join(root, p, p + '_annotated.csv')
        annotations = pd.read_csv(label_file)
        labels = annotations.iloc[:, ann_col_id]
        labels = labels.to_list()
        labels_list.extend(labels)
    
    print("Initial image ids: ", len(image_ids))
    
    new_image_ids, filtered_labels_array = eliminate_confusing_instances(labels_list, image_ids)
    print("After removing confusing instances: ", len(new_image_ids))
    np.savetxt(boundary_list_path, new_image_ids, fmt='%s',  newline='', delimiter='')

    new_image_ids = create_balanced_dataset(filtered_labels_array, new_image_ids)
    print("After balancing the dataset: ", len(new_image_ids))
    np.savetxt(balanced_list_path, new_image_ids, fmt='%s', newline='', delimiter='')

    print("Done Saving!!!")

if __name__ == "__main__":
    main()