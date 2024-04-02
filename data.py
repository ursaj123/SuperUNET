import numpy as np
import os
import PIL
from PIL import Image
import torch
import albumentations as A




def load_image(img_path):
    '''Load an image from a file path and return it as a numpy array'''
    img = Image.open(img_path)
    img = img.convert('RGB')
    img_arr = np.array(img)
    return img_arr


def save_image(img_arr, img_path):
    '''Save a numpy array as an image to a file path'''
    img = Image.fromarray(img_arr)
    img.save(img_path)


def split(images_path, masks_path, train_size=0.9, random_state=42):
    '''Split the images and masks into training and validation sets'''
    num_samples = int(train_size*len(os.listdir(images_path)))
    images_list = np.array(os.listdir(images_path))
    mask_list = np.array(os.listdir(masks_path))

    indices = np.arange(len(os.listdir(images_path)))
    np.random.seed(random_state)
    train_indices = np.random.choice(indices, num_samples, replace=False)
    val_indices = np.array(list(set(indices) - set(train_indices)))
    return list(images_list[train_indices]), list(images_list[val_indices]), list(
        mask_list[train_indices]), list(mask_list[val_indices])


class CustomDataset(torch.utils.data.Dataset):
    '''Custom dataset class for loading images and masks from a directory'''
    def __init__(self, images_dir, masks_dir, images_list, masks_list, transform=None,
                 mode='train', max_samples=1000):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform
        self.mode = mode
        self.max_samples = max_samples

    def __len__(self):
        if self.mode == 'train':
            return max(self.max_samples, len(self.images_list))
        else:
            return len(self.images_list)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            idx = idx%len(self.images_list)
        img_path = os.path.join(self.images_dir, self.images_list[idx])
        mask_path = self.images_list[idx].replace('training.tif', 'manual1.gif')
        mask_path = os.path.join(self.masks_dir, mask_path)

        img_arr = load_image(img_path)
        mask_arr = load_image(mask_path)
        mask_arr = mask_arr[:,:,0]
        mask_arr[mask_arr==255] = 1

        if self.transform is not None:
            transformed = self.transform(image=img_arr, mask=mask_arr)
            img_arr = transformed['image']
            mask_arr = transformed['mask']
        
        return {'img':img_arr, 'mask':mask_arr}
    


