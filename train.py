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


import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

import pickle
from sklearn.metrics import confusion_matrix
from config import get_config


parser = argparse.ArgumentParser(description='SimICL')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run(default: 100)') 
parser.add_argument('--dataset_path', default = './visual_prompt/', type=str) 
parser.add_argument('--train_csv', default='./visual_prompt/train.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str)
parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size (default: 64)') 
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)') 
parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.0005 for AdamW)') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0.6, type=float,
                        help='Masking ratio (percentage of removed patches).')
parser.add_argument('--save_frequency', default=400, type=int,
                        help='save trained model')
parser.add_argument('--cfg', default ='./configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml', type=str,  metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

args = parser.parse_args() 


config = get_config(args)

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print("Folder created")
else:
    print("Folder already exists")



from models import build_model



# dataloader

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
    def __init__(self, input_size=224, mask_patch_size=16, model_patch_size=16, mask_ratio=0.6):
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
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)         
        return mask

class SegmentationDataSet(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, mask_ratio:float, transform=None):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'input/')  
        self.output_path = os.path.join(self.dataset_path, 'gt/')
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
    def __getitem__(self, index: int):
        # Select the sample
        image_filename = self.images_list[index]
        # Load input and target
        image = cv.imread(os.path.join(self.input_path, image_filename),0) # read concatenated input
        gt  = cv.imread(os.path.join(self.output_path, image_filename),0)  # read concatenated ground truth

      
        # add: 3 channel
        image = np.repeat(image[None,...], 3, axis=0).transpose(1, 2, 0)
        gt = np.repeat(gt[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.transform:
            image = self.transform(np.uint8(image))
            gt = self.transform(np.uint8(gt)) 

            
        mask = self.mask_generator()
            
        return image, mask, gt  



# model
model = build_model(config) 
model.to(device)




PATH = args.save_dir + 'latest_model.pth' 
if os.path.exists(PATH): 
    model.load_state_dict(torch.load(PATH))


# Training function

def train_model(dataloader, optimizer, model):
    train_loss = 0
    model.train()
    for images, masks, gts in dataloader: 
        images = images.cuda() 
        masks = masks.cuda()
        gts = gts.cuda()  
        
        loss, rec = model(images, masks, gts) 
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights

        train_loss =+ loss.item()

        
    return train_loss/len(dataloader), rec, masks, model

    
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]), 
        transforms.ToTensor()])

# Create training dataset
training_dataset = SegmentationDataSet(dataset_path = args.dataset_path, df_input = args.train_csv, mask_ratio = args.mask_ratio, transform = train_transform)

# Initialization

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = True)
# Run training and evaluation cycles


print('------------------------------------------------------------------------')
print('Epochs:', args.epochs)
print('Batch size:', args.batch_size)
print('Learning rate:', args.lr)
print('')

with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('Batch size:'+str(args.batch_size)+'Learning rate:'+str(args.lr))

optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
torch.set_grad_enabled(True)

result_path = args.save_dir + "training_result.pickle"


if not os.path.exists(result_path): #if results do not exit
    results = {'train_loss': []}
else:
    results_ = open(result_path,'rb')
    results = pickle.load(results_)
    results_.close()
print(results)

start_epoch = len(results['train_loss'])
for epoch in range(start_epoch, args.epochs):
    train_loss, pred, mask, model= train_model(training_dataloader, optimizer, model)
    # save model
    if epoch % 1 == 0:
        print("(epoch "+str(epoch)+")", 
              "\t"+"train loss: "+str(train_loss))
        torch.save(model.state_dict(),args.save_dir + 'latest_model.pth') 
        with open(args.save_dir + 'training_result.txt', 'a') as f:
            f.write("(epoch "+str(epoch)+")"+ 
                    "\t"+"train loss: "+str(train_loss)+"\n")
    results['train_loss'].append(train_loss)

    pickle.dump(results, open(args.save_dir + "training_result.pickle", "wb")) 
    if (epoch+1) % args.save_frequency == 0:
        torch.save(model.state_dict(),args.save_dir + 'latest_model_'+str(epoch)+'.pth') 






