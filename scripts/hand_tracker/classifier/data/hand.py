import cv2
import numpy as np
import random
import os
import os.path as osp
from torch.utils import data
from PIL import Image
import re
import pandas as pd


class HandDataSet(data.Dataset):
    def __init__(self, root, p_ids, list_path, mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.p_ids = p_ids
        self.img_ids = [i_id.split('/')[-1] for i_id in open(list_path)]
        self.ann_col_id = 42
        self.mean = mean

        label_dict = {}
        self.files = []
        labels_list = []

        self.class_map = {"None": 0, "PICK": 1, "NOT_PICK": 2}

        for p in self.p_ids:
            label_file = osp.join(root, p, p + '_annotated.csv')
            annotations = pd.read_csv(label_file)
            labels = annotations.iloc[:, self.ann_col_id]
            labels_list.extend(labels)
            labels = np.array(labels)
            label_dict[p] = labels
        labels_list = np.array(labels_list)
        np.save('labels_list.npy', labels_list)

        for name in self.img_ids:
            frame_id = name.split('.')[0].split('_')[-1]
            p_id = name.split('.')[0].split('_')[0] + '_' + name.split('.')[0].split('_')[1]
            if p_id in self.p_ids:
                labels = label_dict[p_id]
                lab =  self.class_map[labels[int(frame_id)-1]]
                img_file = p_id+'_frame_'+frame_id
                image_path = osp.join(root, p_id, "Images/%s.png" %(img_file))
                self.files.append({
                    "name": name, 
                    "label": lab, 
                    "image_path": image_path
                    })

    def __len__(self):    
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["image_path"], -1)

        # resize to quarter the size to prevent memory issues maintaining the same aspect ratio (720x1280)
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_CUBIC)

        # convert image to float32 to subtract image mean
        image = np.asarray(image, np.float32)

        # mean subtraction from image
        image -= self.mean

        # transpose to get channels first
        image = image.transpose((2, 0, 1))

        label = np.asarray(datafiles["label"])
        size = image.shape
        name = datafiles["name"]
        return image.copy(), label.copy(), np.array(size), name, index

def main():

    root = osp.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
    list_path = osp.join(root, 'train_file_list.text')
    p_ids = ['P_42', 'P_22', 'P_77']
    hand_dataset = HandDataSet(root, p_ids, list_path)
    # print(hand_dataset.__len__())
    image, label, size, name, index = hand_dataset.__getitem__(500)
    print(np.shape(image), label, size, name, index)


if __name__ == "__main__":
    main()