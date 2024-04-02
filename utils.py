import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import torch
from time import time


def save_config(config_dict, config_path):
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)

def save_model(model, optimizer, epoch, loss, path):
    # we are savinf optimizer and epoch also so that we can resume training from the same point if required 
    # otherwise we will just save the model state_dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }, path)

def load_model(model, optimizer, path):
    # after returning the model, optimizer, epoch and loss, we can resume training from the same point
    # or we can use the model for inference
    # so use model.train() and model.eval() with torch.inference_mode(): as required
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def metrics(pred, target):
    # since this is all about binary segmentation maps, we will be using confusion marix
    # pred shape is (B, num_classes, H, W)
    # target shape is (B, H, W)
    pred = pred.argmax(dim=1) # (B, H, W)
    target = target.long()
    eps = 1e-7
    true_positive = torch.bitwise_and(pred==1, target==1).sum().item() + eps
    true_negative = torch.bitwise_and(pred==0, target==0).sum().item() + eps
    false_positive = torch.bitwise_and(pred==1, target==0).sum().item() + eps
    false_negative = torch.bitwise_and(pred==0, target==1).sum().item() + eps

    all_pixels = true_positive + true_negative + false_positive + false_negative
    accuracy = (true_positive + true_negative)/all_pixels
    positive_predictive_rate = true_positive/(true_positive + false_positive)
    sensitivity = true_positive/(true_positive + false_negative)
    dice_similarity = 2*true_positive/(2*true_positive + false_positive + false_negative)


    return {'acc':accuracy, 'ppr':positive_predictive_rate, 'sens':sensitivity, 'dsc':dice_similarity}


def visualize_segmaps(img_batch, segmap, orig_mask):
    '''
    - this too work in the batched form
    - we will do this with batch sizes of 4
    '''
    batch_size = img_batch.shape[0]
    plt.figure(figsize=(10,10))
    for i in range(batch_size):
        plt.subplot(batch_size, 3, 3*i + 1)
        plt.imshow(img_batch[i].detach().cpu().numpy().transpose(1,2,0))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(batch_size, 3, 3*i + 2)
        plt.imshow(segmap[i].argmax(0).detach().cpu().numpy(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.subplot(batch_size, 3, 3*i + 3)
        plt.imshow(orig_mask[i].detach().cpu().numpy(), cmap='gray')
        plt.title('Original Mask')
        plt.axis('off')

    plt.show()


def save_segmaps(img_batch, segmap, orig_mask, path):
    '''
    - this too work in the batched form
    - we will do this with batch sizes of 4
    '''
    batch_size = img_batch.shape[0]
    for i in range(batch_size):
        plt.figure(figsize=(10,10))
        plt.subplot(1, 3, 1)
        plt.imshow(img_batch[i].detach().cpu().numpy().transpose(1,2,0))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(segmap[i].argmax(0).detach().cpu().numpy(), cmap='gray')
        plt.title('Predicted Mask')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(orig_mask[i].detach().cpu().numpy(), cmap='gray')
        plt.title('Original Mask')
        plt.axis('off')

        plt.savefig(path)
        plt.close()

def time_elapsed(start_time):
    end_time = time()
    elapsed_time = end_time - start_time
    # let us change it to hours minutes and seconds manuaaly
    hours = int(elapsed_time//3600)
    minutes = int((elapsed_time%3600)//60)
    seconds = int(elapsed_time%60)
    elapsed_time = f'{hours}H {minutes}M {seconds}S'
    return elapsed_time