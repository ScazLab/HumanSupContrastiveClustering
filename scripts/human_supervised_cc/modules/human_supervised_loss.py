import torch
import torch.nn as nn
import math
import numpy as np

class HumanSupervisedLoss(nn.Module):
    def __init__(self, device):
        super(HumanSupervisedLoss, self).__init__()
        self.device = device
        self.mean = torch.zeros((1))
        self.similarity_f = nn.CosineSimilarity(dim=1)
        self.old_count = torch.zeros((1)).to(self.device)
        self.old_mean = torch.zeros((1)).to(self.device)
        
    def forward(self, h):
        if h.shape[0]==0:
            return torch.zeros((1)).to(self.device)
        
        h = h.detach().to(self.device)

        # update mean of the cluster with new data points
        new_sum = torch.sum(h, dim=0)
        new_count = h.shape[0]
        self.new_mean = new_sum / new_count

        # calculate the running mean of the cluster
        self.mean = ((self.old_mean * self.old_count) + (self.new_mean * new_count))/(self.old_count + new_count)

        # re-initialize the old count and the old mean
        self.old_count = self.old_count + new_count
        self.old_mean = self.mean

        # cosine similarity loss
        sim = self.similarity_f(self.mean.unsqueeze(0), h)
        loss = -sim.mean()

        return loss