import numpy as np
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torch.utils import data

from hand_tracker.classifier.data.hand import HandDataSet
from train import NUM_EPOCHS, ImageClassifier

from scipy.special import softmax

NUM_CLASSES = 3
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 4
DEVICE = 'cuda'
MODEL_PATH = 'weights/v2/'
NUM_EPOCHS = 25

def main():

    for epoch in range(0, NUM_EPOCHS):
        print("Epoch "+ str(epoch))
        root = os.path.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
        list_path = os.path.join(root, 'all_eval_file_list.text')
        # p_ids = ['P_42', 'P_22', 'P_77']
        p_ids = ['P_1', 'P_3', 'P_5']
        device = DEVICE
        weights_path = os.path.join(MODEL_PATH+'epoch_'+str(epoch)+'.pth')
        dataset = HandDataSet(root, p_ids, list_path, IMG_MEAN)
        dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        X_data, y_data, _, name, _ = next(iter(dataloader))

        net = ImageClassifier()
        net.load_state_dict(torch.load(weights_path))
        net = net.to(device)
        
        correct = 0
        total = 0
        pred_list = []
        correct_c = [0 for c in range(NUM_CLASSES)]
        total_c = [0 for c in range(NUM_CLASSES)]
        acc_c = [0 for c in range(NUM_CLASSES)]
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for dataset in dataloader:
                images, labels, _, name, _ = dataset
                # calculate outputs by running images through the network
                outputs = net(images.to(device))
                labels = labels.to(device)
                outputs = outputs.to(device)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pred_list.append([name, predicted.item(), labels.item()])

                for c in range(NUM_CLASSES):
                    correct_c[c] += ((predicted == labels) * (labels == c)).sum().item() 
                    total_c[c] += (labels == c).sum().item()

            # Class wise accuracy    
            for c in range(NUM_CLASSES):
                acc_c[c] = 100 * correct_c[c]/total_c[c]
            print(acc_c)

            pred_list = np.array(pred_list)
            np.save('preds_'+str(epoch)+'.npy', pred_list)

            print('Accuracy of the network on the test images for all classes: %d %%' % (
            100 * correct / total))

if __name__ == "__main__":
    main()
