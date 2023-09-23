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
num_epochs = 13
batch_size = 100
eta = .0004 # learning rate
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
- Convolution/ReLU w/ 3x3 kernel. Boost to 32 channels
- Max Pool w/ 2x2 kernel and stride 2.
- Convolution/ReLU w/ 3x3 kernel. Boost to 64 channels.
- Convolution/ReLU w/ 3x3 kernel. Maintain 64 channels.
- Max Pool w/ 2x2 kernel and stride 2.
- Convolution/ReLU w/ 3x3 kernel. Boost to 128 channels.
- Convolution/ReLU w/ 3x3 kernel. Boost to 128 channels.
- Max Pool w/ 2x2 kernel and stride 2.
- FLATTEN
- ReLU Linear layer mapping flattened input features to 500 outputs
- ReLU Hidden layer mapping 500 inputs to 500 outputs
- ReLU Hidden layer mapping 500 inputs to 500 outputs
- Score assignment mapping 500 inputs to 10 scores
- Softmax (included in CrossEntropy)

Remarks:
- Image batches have shape [batch_size, channel_count, height, width]
- A single channel image is flattened according to: 
    len(x) = (height*width-kernel_size)/(stride) + 1
- With multiple channels, the flattening becomes:
    len(x) = [(height*width-kernel_size)/(stride) + 1]*channel_count
- Even single datapoints have a "batch dimension" that needs to be preserved
when processing, so DO NOT USE torch.flatten(), because it will eat the batch
dimension and break things.
- Dimensionality reduction only occurs in pooling, padding is used in conv.
layers to induce no dimensionality loss during convolution

Some experimental results:
- Downsizing the kernel to 3x3 from 5x5 significantly improved performance
- Additional hidden layers in the linear phase were expensive and not terribly
helpful; deepening the convolution seemed to work better
'''

class CNN(pl.LightningModule): # inherit from LightningModule

    # Constructor
    def __init__(self):

        super(CNN, self).__init__() # inherit constructor

        # LAYERS

        # 3 color channels, map to 32, use a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, 
                                kernel_size = 3, padding = 2)

        # Dimension reduction w/ 2x2 max pool at stride 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Further feature extraction at 64 channels
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64,
                                kernel_size = 3, padding = 2)
        
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64,
                                kernel_size = 3, padding = 2)
        
        # Dimension reduction w/ 2x2 max pool at stride 2
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Boost channels to 128, maintain 3x3 kernel
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 128, 
                                kernel_size = 3, padding = 2)
        
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 128,
                                kernel_size = 3, padding = 2)
        
        # Dimension reduction w/ 2x2 max pool at stride 2
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)


        # Fully connected layer; in_features has to be hardcoded in.
        self.fc1 = nn.Linear(in_features = 6272, out_features = 500)

        # Additional Fully connected layers to introduce nonlinearity
        self.fc2 = nn.Linear(in_features = 500, out_features = 500)

        self.fc3 = nn.Linear(in_features = 500, out_features = 500)

        # Score assignment
        self.fc4 = nn.Linear(in_features = 500, out_features = 10)




    
    # Forward Pass
    def forward(self, x):

        # Convolutional Layers
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = self.pool2(out)
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.pool3(out)

        # Flattening and Linear Layers. Notice we preserve the batch dimension.
        out = out.reshape(-1, 6272)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out
    
    # Optimizer configuration; SGD is currently considered best practice for CNN
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = eta)
    
    # Training loop configuration
    def training_step(self, batch, batch_idx):

        # Unpacking
        x, y = batch

        # Forward pass
        y_hat = self.forward(x)

        # Loss Computation. nn.CrossEntropyLoss() doesn't work here, not sure why.
        L = F.cross_entropy(y_hat, y)

        # Compute accuracy. Notice logits are on axis 1, axis 0 is the batch axis.  
        acc = y_hat.argmax(dim = 1).eq(y).sum().item() / len(y)

        # Performance Logging
        metrics = {"train_loss": L, "train_acc": acc}
        self.log_dict(metrics)

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

        # Compute accuracy. Notice logits are on axis 1, axis 0 is the batch axis.  
        acc = y_hat.argmax(dim = 1).eq(y).sum().item() / len(y)
        
        # Performance Logging
        metrics = {"val_loss": L, "val_acc": acc}
        self.log_dict(metrics)

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

        # Compute accuracy. Notice logits are on axis 1, axis 0 is the batch axis.  
        acc = y_hat.argmax(dim = 1).eq(y).sum().item() / len(y)

        # Performance Logging
        metrics = {"test_loss": L, "test_acc": acc}
        self.log_dict(metrics)

        # Dictionary formatting
        return {'test_loss': L}
    
    # Testing Loader configuration
    def test_dataloader(self):

        # Declare a test loader, give 4 worker threads
        test_loader = torch.utils.data.DataLoader(dataset = test, 
                                                batch_size = batch_size,
                                                num_workers = 4,
                                                shuffle = False)
        
        return test_loader

    

# %%
# CALL FROM LINE
if __name__ == '__main__':
    trial_flag = input("Debug Mode? [y/n]:")

    # Input Parsing
    if trial_flag == 'y':
        trial_flag = True
    elif trial_flag == 'n':
        trial_flag = False
    else:
        raise ValueError("Answer needs to be 'y' or 'n' ")
    
    # Trainer configuration. We are using CPU training because my hardware is bad.
    trainer = Trainer(max_epochs = num_epochs, fast_dev_run = trial_flag, 
                    accelerator = 'cpu')
    model = CNN()
    trainer.fit(model)
    trainer.validate(model, dataloaders = model.val_dataloader())
    trainer.test(model, dataloaders = model.test_dataloader())
    
    # Save Weights
    torch.save(model.state_dict(), 'lightning_CNN.pth')