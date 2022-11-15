import os
import numpy as np
import torch
import torchvision
from torchvision import transforms, utils
import argparse
from modules import transform, resnet, network, contrastive_loss, human_supervised_loss
from utils import yaml_config_hook, save_model
from torch.utils import data
from data.crops import CropsDataSet
import os.path as osp


def train():
    '''
    Train the human-supervised contrastive clustering network. 
    '''
    
    loss_epoch = 0
    
    for step, (x, x_i, x_j, _, _, names, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to('cuda') # original image
        x_i = x_i.to('cuda') # first augmentation of the original image
        x_j = x_j.to('cuda') # second augmentation of the original image

        # Find which samples in the current batch belong to the human pool
        indexes = list()
        for index, name in enumerate(names):
            if name in human_pool_list:
                indexes.append(index)
    
        # The model takes the original image and its two augmentations and outputs:
        # h, h_i, h_j: image representation from the base encoder
        # z_i, z_j: representation from the instance projection head
        # c_i, c_j: representation from the cluster projection head
        h, h_i, h_j, z_i, z_j, c_i, c_j = model(x, x_i, x_j)

        # Select the embeddings of the human selected samples
        selected_embeddings = h[indexes]    

        # instance loss
        loss_instance = criterion_instance(z_i, z_j)

        # cluster loss
        loss_cluster = criterion_cluster(c_i, c_j)

        # human supervised loss
        loss_human_supervised = criterion_human_supervised(selected_embeddings)

        # aggregate loss
        loss = loss_instance + loss_cluster + loss_human_supervised
        
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(train_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}\t loss_human_supervised: {loss_human_supervised}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()

    root = osp.join(os.path.expanduser("~"), "Crops_Dataset")
    list_path = osp.join(root, 'file_lists')
    human_list_path = osp.join(list_path, 'human_lists')
    human_pool_path = os.path.join(human_list_path, args.human_list_filename)
    human_pool = np.loadtxt(human_pool_path, dtype=str)
    human_pool_list = []

    for name in human_pool:
        img_name = name.split('/')[-1]
        human_pool_list.append(img_name)


        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # generate image augmentations
    s = 1
    color_jitter = torchvision.transforms.ColorJitter(
        0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s 
    )
    gaussian_blur = torchvision.transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5))
    random_resized_crop = torchvision.transforms.RandomResizedCrop(size=args.image_size, scale=(0.5, 1.0))

    transforms = torchvision.transforms.Compose([torchvision.transforms.RandomApply([random_resized_crop, 
                                                                                    color_jitter,
                                                                                    gaussian_blur
                                                                                    ], 
                                                                                    p=0.9)])

     # initialize dataset
    if args.dataset == "crops":
        train_dataset = CropsDataSet(args.dataset_dir, 
                                    args.list_path, 
                                    args.mean,
                                    transforms=transforms)
        class_num = 10  

    train_sampler = None

    # initialize data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=(train_sampler is None),
                                                   drop_last=True,
                                                   num_workers=args.workers,
                                                   sampler=train_sampler,
                                                )


    # initialize model
    res = resnet.get_resnet(args.resnet)
    model = network.Network(res, args.feature_dim, class_num)
    model = model.to('cuda')

    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    criterion_human_supervised = human_supervised_loss.HumanSupervisedLoss(loss_device)
    
    # train
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 100 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}")
    save_model(args, model, optimizer, args.epochs)
