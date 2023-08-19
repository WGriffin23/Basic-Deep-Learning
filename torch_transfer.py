# PyTorch Examples: TRANSFER LEARNING
# Task: Classify Bees and Ants
# We use the pretrained ResNet-18 CNN to learn this task specifically

#%%
# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import numpy as np
import torchvision 
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#%%
# Hyperparameter Initializations
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# Transform Dictionary, several are needed in different contexts
# Things that need transforms are keys, the neeeded transform is a value
data_transforms = {

    'train': transforms.Compose([transforms.RandomResizedCrop(224), 
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(), 
     transforms.Normalize(mean, std)]), 
     
     'val': transforms.Compose([transforms.Resize(256), 
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

                                }


# %%
# Data Loading

data_dir = "data/hymenoptera_data" # directory specification
sets = ('train', 'val') # dataset identifiers
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))}