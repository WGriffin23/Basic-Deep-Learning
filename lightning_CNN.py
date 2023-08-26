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
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt 
import torchvision 
import torchvision.transforms as transforms
import lightning.pytorch as pl
import torch.nn.functional as F
from lightning.pytorch import Trainer

# %%
# HYPERPARAMETERS
num_epochs = 4
batch_size = 100
eta = .001 # learning rate
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
'ship', 'truck')

# Normalizes data and casts it as a tensor.
transform = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))])

#%%
# DATASET IMPORTS

# Downlaod to data file, training dataset, cast to tensor and normalize
train = torchvision.datasets.CIFAR10(root = './data',
                                    train = True,
                                    download = True,
                                    transform = transform)

# Split off a validation dataset from the 50k training images
train, val = random_split(train, lengths = [40000, 10000])

# Download testing dataset to data directory, applying the same transform
test = torchvision.datasets.CIFAR10(root = './data',
                                    train = False,
                                    download = True,
                                    transform = transform)

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

Remarks:
- Image batches have shape [batch_size, channel_count, height, width]
- We don't use padding for this dataset
- Because of the low pixel count, stride is going to be 1 for all conv layers
- Because of the low pixel count, kernel size will stay 5x5 for all conv layers
- A single channel image is flattened according to: 
    len(x) = (height*width-kernel_size)/(stride) + 1
- With multiple channels, the flattening becomes:
    len(x) = [(height*width-kernel_size)/(stride) + 1]*channel_count
- Even single datapoints have a "batch dimension" that needs to be preserved
when processing, so DO NOT USE torch.flatten(), because it will eat the batch
dimension and break things.
'''

class CNN(pl.LightningModule): # inherit from LightningModule

    # Constructor
    def __init__(self):

        super(CNN, self).__init__() # inherit constructor

        # LAYERS

        # 3 color channels, map to 6, use a 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, 
                                kernel_size = 5)

        # map 6 channels to 16 channels, use a 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16,
                                kernel_size = 5)

        # max pool, use a 2x2 kernel and stride of 1
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 1)

        # map 16 channels to 32 channels, use a 5x5 kernel
        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, 
                                kernel_size = 5)
        
        # map 32 channels to 64 channels, use a 5x5 kernel
        self.conv4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 5)

        # max pool, use a 2x2 kernel and stride of 1
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 1)

        # Fully connected layer; in_features has to be hardcoded in
        self.fc1 = nn.Linear(in_features = 12544, out_features = 10)
    
    # Forward Pass
    def forward(self, x):

        # Convolutional Layers
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool1(out)
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool2(out)

        # Flattening and Linear Layers. Notice we preserve the batch dimension.
        out = out.reshape(-1, 12544)
        out = self.fc1(out)

        return out
    
    # Optimizer configuration; SGD is currently considered best practice for CNN
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr = eta)
    
    # Training loop configuration
    def training_step(self, batch, batch_idx):

        # Unpacking
        x, y = batch

        # Forward pass
        y_hat = self.forward(x)

        # Loss Computation. nn.CrossEntropyLoss() doesn't work here, not sure why.
        L = F.cross_entropy(y_hat, y)

        # Dictionary formatting
        return {'loss': L}
    
    # Training loader configuration
    def train_dataloader(self):
        
        # declare batch size, give 4 worker threads to load, and shuffle
        train_loader = torch.utils.data.DataLoader(dataset = train,
                                                   batch_size = batch_size,
                                                   num_workers = 4,
                                                   shuffle = True)
        
        return train_loader
    
    # Validation Loop configuration; Lightning handles no_grad stuff
    def validation_step(self, batch, batch_idx):

        # Unpacking; remember these tensors have a batch dimension
        x, y = batch

        # Forward Pass
        y_hat = self.forward(x)

        # Loss Computation
        L = F.cross_entropy(y_hat, y)

        # Dictionary formatting
        return {'val_loss': L}
    
    def val_dataloader(self):

        # Declare a validation loader, give 4 worker threads, don't allow shuffling
        val_loader = torch.utils.data.DataLoader(dataset = val,
                                                batch_size = batch_size,
                                                num_workers = 4,
                                                shuffle = False)
        
        return val_loader
    
    # Testing Loop configuration; Lightning handles no_grad stuff
    def test_step(self, batch, batch_idx):

        # Unpacking; remember these tensors have a batch dimension
        x, y = batch

        # Forward Pass
        y_hat = self.forward(x)

        # Loss Computation
        L = F.cross_entropy(y_hat, y)

        # Dictionary formatting
        return {'test_loss': L}
    
    # Testing Loader configuration
    def test_dataloader(self):

        # Declare a test loader, give 4 worker threads
        val_loader = torch.utils.data.DataLoader(dataset = val, 
                                                batch_size = batch_size,
                                                num_workers = 4,
                                                shuffle = False)
        
        return val_loader

    

# %%
# CALL FROM LINE
if __name__ == '__main__':
    trainer = Trainer(max_epochs = num_epochs, fast_dev_run = True, 
                    accelerator = 'cpu')
    model = CNN()
    trainer.fit(model)
    trainer.test(model, dataloaders = model.test_dataloader())