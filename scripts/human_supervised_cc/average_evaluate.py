import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import softmax
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from modules import transform, resnet, network, contrastive_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from data.crops import CropsDataSet
import glob

from PIL import Image
import pdb
from matplotlib import cm

from utils.yaml_config_hook import yaml_config_hook
import itertools
from itertools import cycle
import cv2
from matplotlib import gridspec
import math
from sklearn.manifold import TSNE
from collections import Counter
import json

def inference(model, train_loader, human_image_list, subcat_name, human_list):
    metric_dict = {}
    similarity_f = nn.CosineSimilarity(dim=2)
    average_sim_list = []
    subcat_list = []
    with torch.no_grad():
        for step, (_ , x_i1, x_i2, _, subcat_j , name_j, _) in enumerate(train_loader):
            x_j = human_image_list
            x_j = torch.permute(x_j, (0, 3, 1, 2))
            x_i = x_i1.to(args.device)
            x_j = x_j.to(args.device)

            # the paper takes the representations of the encoder and not the MLP during evaluation
            x = torch.cat((x_i, x_j))
            h, h_i, h_j, z_i, z_j, c_i, c_j = model(x, x, x) 

            sim = similarity_f(h.unsqueeze(1), h.unsqueeze(0))
            sims = sim[0][1:].cpu().detach().numpy()
            sims = sorted(sims)
            #top_5 = sims[-1]

            average_sim = sims[-1] # top 1 # np.average(sims)
            # average_sim = np.average(sims[-5:]) # top 5
            # average_sim = sims[-3] # 3rd best
            average_sim_list.append(average_sim)
            subcat_list.append(subcat_j[0])

        tp, fp, tn, fn, precision, recall, f1, threshold, subcat = calculate_metrics(average_sim_list, subcat_list, subcat_name)
        metric_dict["tp"] = tp
        metric_dict["fp"] = fp
        metric_dict["tn"] = tn
        metric_dict["fn"] = fn
        metric_dict["precision"] = precision
        metric_dict["recall"] = recall
        metric_dict["f1"] = f1
        metric_dict["threshold"] = threshold
        metric_dict["subcat"] = subcat
        return metric_dict

def calculate_metrics(average_sim_list, subcat_list, subcat_name):
    '''
    Calculate evaluation metrics given the list of similarities
    '''
    best_f1 = -1
    best_threshold = -1
    for t in np.arange(0.1, 1.0, 0.02):
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for (average_sim, subcat) in zip(average_sim_list, subcat_list):
            if subcat == subcat_name:
                if average_sim >= t:
                    tp+=1
                else:
                    fn+=1
            if subcat != subcat_name:
                if average_sim >= t:
                    fp+=1
                else:
                    tn+=1

        precision = tp/(tp+fp)            
        recall = tp/(tp+fn)            
        f1 = (2 * precision * recall)/(precision + recall)

        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_tp = tp
            best_tn = tn
            best_fp = fp
            best_fn = fn
            best_f1 = f1
            best_threshold = t

    print("****************************")
    print("True Positives: ", best_tp)
    print("False Positives: ", best_fp)
    print("True Negatives", best_tn)
    print("False Negatives", best_fn)
    print("Precision: ", best_precision)
    print("Recall", best_recall)
    print("Best F1: ", best_f1, "Best Threshold: ", best_threshold)
    print("****************************")
    return best_tp, best_fp, best_tn, best_fn, best_precision, best_recall, best_f1, best_threshold, subcat_name

def plot_metrics(human_metric_dict):
    subcat_f1_mean_dict = {}
    subcat_f1_std_dict = {}
    subcats = ["Tins", "Crushed_Cans", "Intact_Cans",
           "Brown_Cardboard", "Coated_Cardboard", 
           "Trays", "Colored_Bottles",
           "Crushed_Bottles", "Intact_Bottles",
           "One_Gallon", "Half_Gallon"]
    for s in subcats:
        f1_list = []
        for filename, metrics in human_metric_dict.items():
            if s == metrics["subcat"]:
                f1_list.append(metrics["f1"])
        subcat_f1_mean_dict[s] = np.mean(f1_list)
        subcat_f1_std_dict[s] = np.std(f1_list)

    return subcat_f1_mean_dict, subcat_f1_std_dict

def main(gpu, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

     # initialize dataset
    if args.dataset == "crops":
        train_dataset = CropsDataSet(args.dataset_dir, 
                                    args.list_path, 
                                    args.mean,
                                    transforms=None)
        class_num = 10 

    train_sampler = None

    # initialize data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=1, # so that crops are fed one by one to be compared to all the human crops
                                                   shuffle=(train_sampler is None),
                                                   drop_last=True,
                                                   num_workers=args.workers,
                                                   sampler=train_sampler,
                                                )


    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')
    model.eval()

    model_fp = os.path.join(
    args.model_path, args.dataset, args.exp_id, "checkpoint_{}.tar".format(args.eval_epoch)
    )
    print(model_fp)
    model.load_state_dict(torch.load(model_fp, map_location=args.device.type)['net'])
    
    human_metric_dict = {}
    data_root = os.path.join(os.path.expanduser("~"), "Crops_Dataset")
    human_text_file_root = os.path.join(data_root, 'file_lists/human_lists')
    
    for i, human_txt_file in enumerate(sorted(os.listdir(human_text_file_root))):
        print(human_txt_file)
        human_data = np.loadtxt(os.path.join(human_text_file_root, human_txt_file), dtype=str)
        human_image_list = []
        human_name_list = []
        for cd in human_data:
            human_name_list.append(cd)
            image = Image.open(cd)
            image = image.resize((128, 128))
            image = np.asarray(image, np.float32)
            human_image_list.append(image)
            subcat = cd.split('/')[4]
        human_image_list = torch.as_tensor(human_image_list)

        metric_dict = inference(model, train_loader,  human_image_list, subcat, human_txt_file.split('.')[0])
        human_metric_dict[human_txt_file] = metric_dict
    
    final_metric_dict = {}
    subcat_f1_mean_dict, subcat_f1_std_dict = plot_metrics(human_metric_dict)
    final_metric_dict["mean_f1"] = subcat_f1_mean_dict
    final_metric_dict["std_f1"] = subcat_f1_std_dict

    json_save_file = open(os.path.join("results/", args.metric_save_json), "w")
    json.dump(final_metric_dict, json_save_file, indent=2)
    json_save_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = 1


    main(0, args)