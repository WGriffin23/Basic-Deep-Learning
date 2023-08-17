# PyTorch Examples: TRANSFER LEARNING
# Task: Classify Bees and Ants

#%%
# Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import numpy as np
import torchvision 
from torchvision import datasts, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

#%%