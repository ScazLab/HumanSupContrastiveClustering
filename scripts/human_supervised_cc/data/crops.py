import cv2
import numpy as np
import random
import os
import os.path as osp
from torch.utils import data
from PIL import Image
import re
import pandas as pd
import matplotlib.pyplot as plt


class CropsDataSet(data.Dataset):
    def __init__(self, root, list_path, mean=(128, 128, 128), transforms=None):
        self.root = root
        self.list_path = list_path
        self.transforms = transforms
        self.list_path = os.path.join(list_path, 'all.txt')
        # self.human_list_path = human_list_path
        # self.human_pool = os.path.join(self.human_list_path, 'Human_Crushed_Cans_1.txt')
        self.mean = mean
        self.data = []

        examples = np.loadtxt(self.list_path, dtype=str)
        for e in examples:
            subcat = e.split('/')[-2]
            name = e.split('/')[-1]
            self.data.append({"subcat": subcat,
                                    "name": name,
                                    "image_path": e 
                                        })
    def __len__(self):  
        return len(self.data)

    def __getitem__(self, index):
        datafiles = self.data[index]
        image = Image.open(datafiles["image_path"])
        image = image.resize((128, 128))
        if self.transforms is not None:
            image1 = self.transforms(image)
            image2 = self.transforms(image)         
        else:
            image1 = image2 = image
        
        image = np.array(image)
        image1 = np.array(image1)
        image2 = np.array(image2)
        
        image = np.asarray(image, np.float32)
        image1 = np.asarray(image1, np.float32)
        image2 = np.asarray(image2, np.float32)
        
        # mean subtraction from image
        # image -= self.mean
        image = image.transpose((2, 0, 1))
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        size = np.shape(image)
        subcat = datafiles["subcat"]
        name = datafiles["name"]
     
        return image, image1.copy(), image2.copy(), np.array(size), subcat, name, index


def main():
    root = osp.join(os.path.expanduser("~"), "Crops_Dataset")
    list_path = osp.join(root, 'file_lists')
    human_list_path = osp.join(list_path, 'human_lists')
    dataset = CropsDataSet(root, list_path)
    print(dataset.__len__())
    image1, image2, size, subcat, name, index = dataset.__getitem__(1000)
    print(image1.shape, image2.shape, size, subcat, name, index)


if __name__ == "__main__":
    main()