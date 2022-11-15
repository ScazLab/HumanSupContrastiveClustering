import numpy as np
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
from torch.utils import data

from hand_tracker.classifier.data.hand import HandDataSet
# from data.hand import HandDataSet
from scipy.special import softmax

NUM_CLASSES = 3
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
BATCH_SIZE = 4
DEVICE = 'cuda'
MODEL_PATH = 'weights/v2/'
NUM_EPOCHS = 25

class ImageClassifier(nn.Module):
    def __init__(self):
       super(ImageClassifier,self).__init__()
       self.net = models.resnet18(pretrained=True, progress=True)
       self.final_layer = nn.Linear(1000, NUM_CLASSES)
       self.log_softmax=nn.LogSoftmax()   

    def forward(self,x):
       x1 = self.net(x)
       y = self.final_layer(x1)
       return y

def main():

    root = os.path.join(os.path.expanduser("~"), "hand_pose_classification_dataset")
    list_path = os.path.join(root, 'balanced_train_file_list.text')
    p_ids = ['P_7', 'P_20', 'P_22', 'P_23', 'P_24', 'P_25', 'P_27', 'P_42', 'P_74', 'P_77']
    device = DEVICE

    classifier = ImageClassifier()

    dataset = HandDataSet(root, p_ids, list_path, IMG_MEAN)
    dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    X_data, y_data, _, name, _ = next(iter(dataloader))
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, images in enumerate(dataloader, 0):
            # get the inputs; images is a list of [inputs, labels]
           
            inputs, labels, _, _, _ = images

            # zero the parameter gradients
            optimizer.zero_grad()
            classifier = classifier.to(device)
            # forward + backward + optimize
            outputs = classifier(inputs.to(device))
            outputs = outputs.to(device)
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        weight_save_path = os.path.join(MODEL_PATH, 'epoch_'+str(epoch)+'.pth')
        torch.save(classifier.state_dict(), weight_save_path)
        print("Weight saved for epoch ", epoch)

    print('Finished Training')

    

    

if __name__ == "__main__":
    main()
