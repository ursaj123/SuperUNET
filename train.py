import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import tqdm
import argparse
import wandb
from time import time
import datetime
import sys
import json

import torch
import torch.nn as nn
import torchvision
import PIL
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from utils import *


# scheduler
def Cosine_Annealing_LinearWarmup_Scheduler(optimizer, init_lr, max_lr, 
                                            warmup_steps, global_step, total_steps):
    '''
    - init_lr is the initial learning rate
    - max_lr is the the learning rate which will achieved after the initial warmup
    - warmup_steps is the number of steps taken to reach the max_lr from init_lr
    - global_step is the current step number in the epoch
    - total_steps = 

    '''
    
    if global_step < warmup_steps:
        lr = init_lr + (max_lr - init_lr)*global_step/warmup_steps
    else:
        lr = init_lr + 0.5*(max_lr - init_lr)*(1 + np.cos((global_step - warmup_steps)/(total_steps - warmup_steps)*np.pi))


    optimizer.param_groups[0]['lr'] = lr


def segment(img, model, num_classes=2, device='cuda'):
    '''
    - we can multiple images at the same time, thus it works in the batched format
    '''
    model.eval()
    with torch.inference_mode():
        img = img.type(torch.float32).to(device)
        # orginally we segmented the images of size 48x48, but
        # now we are getting the images of size 528x528, so we will use sliding window technique
        # to generate the segmentation maps
        segmaps = torch.zeros((img.shape[0], num_classes, 528, 528)).to(device)
        # we will generate the segmentation maps for 11x11 patches of 48x48
        iters = tqdm.tqdm(range(121), 'generating patches', colour='yellow')
        for i in iters: # iterating over number of patches
            row, col = (i//11)*48, (i%11)*48
            img_patch = img[:,:,row:row+48, col:col+48]
            segmaps_patch = model(img_patch)
            segmaps[:,:,row:row+48, col:col+48] = segmaps_patch

        
        return segmaps



def train(model, optimizer, loss_fn, train_loader, val_loader, config, 
          scheduler=None, scaler = None):
    start_time = time()
    global_step = 0 # to be used for learning rate scheduling
    best_loss = np.inf # to be used for model checkpointing, to be stored for lowest train loss
    train_performance_metrics = [] # to be used for training logs
    val_performance_metrics = [] # to be used for validation logs
    prev_val_loss = np.inf # to be used for early stopping, applied on train loss only
    counter = 0 # to be used for early stopping, applied on train loss only

    for epoch in range(config['epochs']):
        print(f'Epoch: {epoch+1}/{config["epochs"]}')
        model.train()
        # setting up the learning rate for this step
        if scheduler is not None:
            Cosine_Annealing_LinearWarmup_Scheduler(optimizer, config['init_lr'], config['lr'], 
                                                    config['warmup_steps'], global_step, 
                                                    config['total_steps'])
        
        epoch_train_performance_metrics = {'loss':0.0, 'acc':0.0, 'ppr':0.0, 'sens':0.0, 'dsc':0.0}
        epoch_val_performance_metrics = {'loss':0.0, 'acc':0.0, 'ppr':0.0, 'sens':0.0, 'dsc':0.0}

        batches = tqdm.tqdm(enumerate(train_loader, 0), 'training', total=len(train_loader), colour='green')
        for i, batch in batches:
            optimizer.zero_grad()
            imgs, masks = batch['img'].type(torch.float32).to(config['device']), batch['mask'].type(torch.long).to(config['device'])

            # forward pass
            op = model(imgs)
            loss = loss_fn(op, masks)
            epoch_train_performance_metrics['loss'] += loss.item()

            # backward pass
            if scaler is not None:
                # using automated mixed precision
                scaler.scale(loss).backward()
                # gradient clipping to avoid exploding gradients
                if config['gradient_clip'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # without automated mixed precision
                loss.backward()
                # gradient clipping to avoid exploding gradients
                if config['gradient_clip'] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                optimizer.step()

            # logging
            global_step += 1
            batches.set_postfix({'loss':loss.item()})
            batch_metrics = metrics(op, masks)
            for key in batch_metrics.keys():
                epoch_train_performance_metrics[key] += batch_metrics[key]

            # now model checkpointing (model checkpointing is done on the global_step)
            # dividing the loss by number of images to get the average loss per image of the batch
            # as the batch size can be different
            if loss.item()/imgs.shape[0] < best_loss:
                best_loss = loss.item()/imgs.shape[0]
                save_model(model, optimizer, epoch, loss, config['model_ckpt_path'])
            
            # early stopping based on train loss if not improved from past 5 epochs or given number of early stopping epochs
            if config['early_stopping'] is not None and loss.item()/imgs.shape[0] > prev_val_loss:
                counter += 1
                if counter == config['early_stopping']:
                    counter = -1
                    break
            else:
                counter = 0

        if counter == -1:
            # now goes the early stopping
            print(f"-------/nEarly Stopping as loss not improved from past {config['early_stopping']} epochs/n-------")
            break
        

        # now all the stuff related to training has been done
        # now we will do the validation part, model.eval() and with torch.no_inference() has 
        # alredy been written in the segment function
        for i, batch in enumerate(val_loader):
            imgs, masks = batch['img'].type(torch.float32).to(config['device']), batch['mask'].type(torch.long).to(config['device'])
            op = segment(imgs, model, config['num_classes'], config['device'])
            loss = loss_fn(op, masks)
            epoch_val_performance_metrics['loss'] += loss.item()
            batch_metrics = metrics(op, masks)

            for key in batch_metrics.keys():
                epoch_val_performance_metrics[key] += batch_metrics[key]

            # now we can log images for visual purposes
            segmap_path = os.path.join(config['test_images_dir'], f'segmap_{epoch}_{i}.png')
            save_segmaps(batch['img'].to(config['device']), op, masks, segmap_path)
            


        # logging the epoch performance metrics
        for key in epoch_train_performance_metrics.keys():
            epoch_train_performance_metrics[key] /= len(train_loader)

        for key in epoch_val_performance_metrics.keys():
            epoch_val_performance_metrics[key] /= len(val_loader)

        train_performance_metrics.append(epoch_train_performance_metrics)
        val_performance_metrics.append(epoch_val_performance_metrics)

        # now logging both of the losses as a csv file
        logs = pd.DataFrame({'train_loss':[x['loss'] for x in train_performance_metrics],
                             'train_acc':[x['acc'] for x in train_performance_metrics],
                             'train_ppr':[x['ppr'] for x in train_performance_metrics],
                             'train_sens':[x['sens'] for x in train_performance_metrics],
                             'train_dsc':[x['dsc'] for x in train_performance_metrics],
                             'val_loss':[x['loss'] for x in val_performance_metrics],
                             'val_acc':[x['acc'] for x in val_performance_metrics],
                             'val_ppr':[x['ppr'] for x in val_performance_metrics],
                             'val_sens':[x['sens'] for x in val_performance_metrics],
                             'val_dsc':[x['dsc'] for x in val_performance_metrics]})
        logs.to_csv(config['logs_path'], index=False)


        print(f'Time Elapsed: {time_elapsed(start_time)}')

def main():
    # let us first of all build all of the config
    # let us first set up all of the paths for the expts
    all_expts = os.makedirs('all_expts', exist_ok=True)
    from datetime import datetime
    t_init = str(datetime.now())
    t_init = t_init.replace(' ', '_')
    t_init = t_init.replace(':', '_')
    t_init = t_init.replace('-', '_')
    t_init = t_init.split('.')[0]
    expt_name = f'expt_{t_init}'

    # now we will create a directory for this expt
    expt_dir = os.path.join('all_expts', expt_name)
    os.makedirs(expt_dir, exist_ok=True)

    folders_to_create = ['models', 'logs', 'test_images']
    for folder in folders_to_create:
        folder = os.path.join(expt_dir, folder)
        os.makedirs(folder, exist_ok=True)

    model_ckpt_dir = os.path.join(expt_dir, 'models')
    model_ckpt_path = os.path.join(model_ckpt_dir, 'best_model.pth')
    logs_dir = os.path.join(expt_dir, 'logs')
    logs_path = os.path.join(logs_dir, 'logs.csv')
    test_images_dir = os.path.join(expt_dir, 'test_images')
    conifg_path = os.path.join(expt_dir, 'config.json')

    config = {
    # checkpoint paths
    'model_ckpt_path':model_ckpt_path,
    'logs_path':logs_path,
    'test_images_dir':test_images_dir,

    # data args
    'images_path':'data/DRIVE',
    'split_ratio':0.9,
    'num_workers':2,
    'shuffle':True,


    # model args
    'patch_size':48,
    'num_classes':2,
    'crop_size':528,
    'weights':[1,4], # class 1 is 4 times more important to predict than class 0


    # data args and model ars
    'train_batch_size':32,
    'val_batch_size':4,
    'lr':1e-3,
    'weight_decay':1e-5,
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs':20,
    'scheduler':None, # None, 'cosine_annealing_linear_warmup'
    'init_lr':0.0, # required in scheduler in case of warm starts
    'warmup_steps':10, # required in scheduler in case of warm starts 
    'total_steps':1000, # required in scheduler in case of warm starts
    'amp':False, # automated mixed precision
    'gradient_clip':5.0,
    'early_stopping':5 # if applied it will be applied to model only
    }

    # we will use parsers here for getting training specific arguments to update the above base config
    def map_arg_as_int(arg):
        return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=str, default='data/DRIVE', help='path to the images directory')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='ratio to split the data into training and validation')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to shuffle the data or not')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=4, help='batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for the optimizer')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train the model')
    parser.add_argument('--amp', type=bool, default=False, help='whether to use automated mixed precision or not')
    parser.add_argument('--gradient_clip', type=float, default=5.0, help='gradient clipping value')
    parser.add_argument('--early_stopping', type=int, default=5, help='number of epochs to wait for early stopping')
    parser.add_argument('--device', type=str, default='cuda', help='device to run the model on')
    parser.add_argument('--patch_size', type=int, default=48, help='size of the patch to be used for training')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes to predict')
    parser.add_argument('--crop_size', type=int, default=528, help='size of the image to be cropped to')
    parser.add_argument('--weights', type=map_arg_as_int, default=[1,4], help='weights for loss function to handle class imbalance')
    parser.add_argument('--scheduler', type=str, default=None, help='scheduler to be used for training')
    parser.add_argument('--init_lr', type=float, default=0.0, help='initial learning rate for the scheduler')
    parser.add_argument('--warmup_steps', type=int, default=10, help='number of steps to warmup the learning rate')
    parser.add_argument('--total_steps', type=int, default=1000, help='total number of steps for the scheduler')

    args = parser.parse_args()
    for key in vars(args).keys():
        config[key] = vars(args)[key]


    from utils import save_config
    print(f'-----------------Saving the config file-----------------')
    save_config(config, config_path=conifg_path)
    

    # let us first of all create the datasetsand dataloaders
    print(f'-----------------Creating Datasets and Dataloaders-----------------')
    from data import split, CustomDataset
    print(f'-----------------Splitting the data-----------------')
    images_path = os.path.join(config['images_path'], 'training', 'images')
    masks_path = os.path.join(config['images_path'], 'training', '1st_manual')
    train_images, val_images, train_masks, val_masks = split(
        images_path, masks_path, config['split_ratio'])
    
    print(f'-----------------Creating the datasets-----------------')
    # applying data augmentation also
    train_transform = A.Compose([
        A.Resize(width=528, height=528), # changing to 528x528
        A.RandomCrop(width=48, height=48), # randomly cropping patches of 48x48
        A.Rotate(limit=20, p=0.5), # geometric transformation
        A.GaussNoise(p=0.5), # adding noise
        A.GaussianBlur(p=0.5), # smoothing
        A.RandomBrightnessContrast(p=0.5), # changing perturbations
        ToTensorV2() # converting to tensor
    ])

    val_transform = A.Compose([
        A.Resize(width=528, height=528),
        ToTensorV2()
    ])

    train_dataset = CustomDataset(images_dir=images_path, masks_dir=masks_path,
                                  images_list=train_images, masks_list=train_masks,
                                  transform=train_transform, mode='train')
    
    val_dataset = CustomDataset(images_dir=images_path, masks_dir=masks_path,
                                images_list=val_images, masks_list=val_masks,
                                transform=val_transform, mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'],
                              shuffle=config['shuffle'], num_workers=config['num_workers'])
    
    val_loader = DataLoader(val_dataset, batch_size=config['val_batch_size'],
                            shuffle=False, num_workers=config['num_workers'])

    print(f'-----------------Datasets and Dataloaders created---------------')

    # let us now create the model
    print(f'-----------------Setting up model, optimizer, loss function etc...-----------------')
    from network_arch import SuperUNET
    model =  SuperUNET(num_classes=config['num_classes']).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters()
                                 , lr=config['lr'], weight_decay=config['weight_decay'])
    
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config['weights']).type(
                                    torch.float32).to(config['device']))
    
    # let us now set up the scaler for mixed precision training
    scaler = None
    if config['amp']:
        scaler = torch.cuda.amp.GradScaler()
    
    print(f'--------------Everything set up, let us now start training---------------')

    # let us now start training
    train(model = model, optimizer=optimizer, 
          loss_fn = loss_fn,
          train_loader=train_loader, val_loader=val_loader, config=config,
          scheduler=config['scheduler'], scaler=scaler)

    print(f'-----------------Training Done-----------------')
    print(f'-----------------Refer to notebbok for loading checkpoints and running it for inferences on new images-----------------')

if __name__ == '__main__':
    main()


# run command 
# python train.py --images_path data/DRIVE
#     --split_ratio 0.7 
#     --lr 1e-3 
#     --weight_decay 1e-5
#     --epochs 20
#     --amp False 
#     --gradient_clip 5.0
#     --early_stopping 5
#     --device cuda 
#     --weights 1 4 
#     --scheduler None 
    
# rest will be used as default arguments