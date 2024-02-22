#!/usr/bin/env python
# coding: utf-8


# If there's a GPU available...
import torch

if torch.cuda.is_available():
    device = "cuda"
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print('We will use CPU')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from skimage.io import imread
from skimage.transform import resize

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import cv2 as cv

import glob

import pickle
import re
from sklearn.metrics import confusion_matrix
from config import get_config
from models import build_model


parser = argparse.ArgumentParser(description='SimICL')
parser.add_argument('--dataset_path', default = './visual_prompt/', type=str) 
parser.add_argument('--test_csv', default='./visual_prompt/val_test.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str) 
parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
parser.add_argument('--save_figure', default=False, type=bool,
                        help='save test prediction')
parser.add_argument('--model_name', default='latest_model.pth', type=str,
                        help='model name')
parser.add_argument('--cfg', default ='./configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml', type=str,  metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
args = parser.parse_args() 

config = get_config(args)


def evaluation_metrics(y_true, y_pred, smooth = 0.0001):
    #y_true = y_true.detach().numpy()
    #y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(dice, jaccard)



import math
import random
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as T
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate
from torchvision.datasets import ImageFolder
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=32, model_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1) #2D expand by 2(scale: 32//16)
        
        return mask

class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path, df_input, mask_ratio,transform=None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'input_test/')
        self.output_path = os.path.join(self.dataset_path, 'gt_test/')
        self.df = pd.read_csv(df_input)
        self.images_list = list(self.df['filename'])
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transform = transform
        
        if config.MODEL.TYPE == 'swin':
            model_patch_size=config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
         
        self.mask_generator = MaskGenerator(
            input_size=224,
            mask_patch_size=16, 
            model_patch_size=16,
            mask_ratio=mask_ratio,
        )
            
    def __len__(self):
        return len(self.images_list)
    def __getitem__(self, index):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input and target
        image = cv.imread(os.path.join(self.input_path, image_filename),0)
        gt  = cv.imread(os.path.join(self.output_path, image_filename),0)         
       
        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        gt = np.repeat(gt[None,...], 3, axis=0).transpose(1, 2, 0) 
        # add transform
        if self.transform:
            image = self.transform(np.uint8(image))
            gt = self.transform(np.uint8(gt))

            
        mask = self.mask_generator()
 
        return image, mask, gt, image_filename



# model
model = build_model(config)
model.to(device)

PATH = args.save_dir + args.model_name
if os.path.exists(PATH):
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
else:
    print('no pretrained model found')



test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]), 
        transforms.ToTensor()])

# Create training dataset
test_dataset = SegmentationDataSet(dataset_path = args.dataset_path, 
                                       df_input = args.test_csv, mask_ratio = args.mask_ratio, transform = train_transform)


test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False)


save_images = args.save_dir + 'test/'

if not os.path.exists(save_images):
    os.makedirs(save_images)
    print("Folder created")
else:
    print("Folder already exists")

def eval_model(dataloader, model, save_path):
    eval_loss = 0
    model.eval()
    dices = []
    jaccards = []

    with torch.no_grad():
        i = 0
        for images, masks, gts, filename in dataloader:
            i += 1
            images = images.cuda() 
            masks = masks.cuda()  
            gts = gts.cuda()
            loss, rec = model(images, masks, gts) 
            images = images.cpu() 
            masks = masks.cpu()  
            gts = gts.cpu()
            rec = rec.cpu() 
            eval_loss =+ loss.item()       


            masks = masks.numpy()
            mask = masks.reshape(14, 14)
            mask = np.repeat(mask, 16, axis=0).repeat(16, axis=1) 
            
            pred1 = rec.numpy()
            pred1 = (pred1 - np.min(pred1)) / np.ptp(pred1)
            
            
            # calculate dice and jaccard
            gt_temp = gts.numpy() #0-1, (1, 3, 224, 224)
            gt_temp = gt_temp[0,:, 112:,112:]
            gt_temp[gt_temp>0.5] = 1
            gt_temp[gt_temp<=0.5] = 0
            pred_temp = pred1[0,:, 112:,112:]
            pred_temp[pred_temp>0.5] = 1
            pred_temp[pred_temp<=0.5] = 0
            dice, jaccard = evaluation_metrics(gt_temp,pred_temp)
            dices.append(dice)
            jaccards.append(jaccard)


            if i%100 == 0 and args.save_figure:  
                cv.imwrite(save_path+filename[0], pred1[0].transpose(1,2,0)*255)
    return dices, jaccards


dices, jaccards= eval_model(training_dataloader, model, save_images) 
print(sum(dices)/len(dices), sum(jaccards)/len(jaccards))

