# PyTorch Tutorial: LIGHTNING WRAPPER
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

#%%
# HYPERPARAMETERS
input_size = 784 # 28x28 images
hidden_size = 500
class_count = 10
num_epochs = 2
batch_size = 100
eta = .001 # learning rate

# %%
# CLASS DESIGN

'''
LightningModule is very similar to nn.module but with additional steps
abstracted away and made easier within the class design
'''
class ANN(pl.LightningModule): # inherit from LightningModule instead
    
    # constructor
    def __init__(self, input_size, hidden_size, class_count):
        super(ANN, self).__init__() # we still inherit the constructor 
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, class_count)

    # forward pass 
    def forward(self, x):

        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        # return raw scores, no SoftMax due to CrossEntropyLoss
        return out

    # optimizer setup, just return the torch optimizer you want
    def configure_optimizers(self): 

        # notice we use self.parameters() since we are within the class
        return torch.optim.Adam(self.parameters(), lr = eta)

    # training updates declaration, handles unpacking batches, zero_grad, and steps
    def training_step(self, batch, batch_idx):

        # unpacking
        images, labels = batch 
        images = images.reshape(-1, 28*28) # flatten images

        # forward pass
        y_hat = self(images) 

        # loss computation
        loss = F.cross_entropy(y_hat, labels)

        # dictionary formatting for Lightning; very particular about key
        return {'loss': loss}
    
    def train_dataloader(self):

        # import training dataset, cast as tensor, download to disk
        train = torchvision.datasets.MNIST(root = "./data", 
                                            train = True,
                                            transform = transforms.ToTensor(),
                                            download = True)

        # initialize a loader at our desired batch size, shuffle for randomness
        train_loader = torch.utils.data.DataLoader(dataset = train,
                                                    batch_size = batch_size,
                                                    num_workers = 4,
                                                    shuffle = True)

        return train_loader
    
    def validation_step(self, batch, batch_idx):

        # unpacking
        images, labels = batch
        images = images.reshape(-1, 28*28) # flatten image; preserve channels

        # forward pass
        y_hat = self(images)

        # loss computation
        loss = F.cross_entropy(y_hat, labels)

        # special dic formatting for Lightning
        return {'val_loss': loss}

    def val_dataloader(self):
        
        # pull val set from testing data, cast it as a tensor
        validation = torchvision.datasets.MNIST(root = './data',
                                                train = False,
                                                transform = transforms.ToTensor())
        
        # initialize loader using the same batch size; no need to shuffle, use 4 worker
        val_loader = torch.utils.data.DataLoader(dataset = validation,
                                                batch_size = batch_size,
                                                num_workers = 4,
                                                shuffle = False)
        
        return val_loader
# %%
# CALL FROM SYSTEM

if __name__ == '__main__':
    trainer = Trainer(max_epochs = num_epochs, fast_dev_run = False) 
    model = ANN(input_size, hidden_size, class_count)
    trainer.fit(model)