# PyTorch Lightning CNN
'''
- Dataset is CIFAR-10
- 60k 32x32 color images (3 channels)
- 10 classes, 6k images per class
- 1k testing images per class
- 5k training images per class
'''

#%%
# IMPORTS

import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import torchvision 
import torchvision.transforms as transforms
import lightning.pytorch as pl
import torch.nn.functional as F
from lightning.pytorch import Trainer

# %%
# HYPERPARAMETERS

num_epochs = 4
batch_size = 96
eta = .001 # learning rate
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
'ship', 'truck')

# %%
# CLASS DESIGN
'''
Layer Guide:
- Convolution/ReLU
- Convolution/ReLU
- Max Pool
- Convolution/ReLU
- Convolution/ReLU
- Max Pool
- Fully Connected
- SoftMax **included in nn.CrossEntropyLoss()
'''

class CNN(pl.LightningModule): # inherit from LightningModule

    # Constructor
    def __init__(self):

        super(CNN, self).__init__() # inherit constructor
        