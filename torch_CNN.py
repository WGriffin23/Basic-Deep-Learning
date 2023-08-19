# PyTorch Example: CONVOLUTIONAL NEURAL NETWORK
'''
Remarks:
- CIFAR-10 is a famous 10 class machine learning dataset
- Data are 32x32 color images 
- There are 6000 images per class, for a total of 60,000 images
- Training and Testing datasets are pre-curated at a 5:1 ratio
'''

#%%
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# %%
# Hyperparameters
num_epochs = 4
batch_size = 96
eta = .001 # learning rate

# %%
# Data Loading

# Cast data to a tensor and standardize the data
transform = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))])

# Loaders; torchvision already caches these as Dataset objects
train = torchvision.datasets.CIFAR10(root = './data', train = True,
download = True, transform = transform)

test = torchvision.datasets.CIFAR10(root = './data', train = False,
download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size,
shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size,
shuffle = False)

# Hard code classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
'ship', 'truck')

# %%
# Model Design

'''
LAYER GUIDE:
- Convolution/ReLU
- Convolution/ReLU
- MaxPool
- Convolution/ReLU
- Convolution/ReLU
- MaxPool
- Fully Connected
- Softmax **included in nn.CrossEntropyLoss()**
'''
class CNN(nn.Module):

    def __init__(self):

        # Always need class inheritence
        super(CNN, self).__init__()

        # Layers

        # 3 color channels, map to 6 channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)

        # 6 channels from first layer, map to 16 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

        # 2x2 pooling kernel, stride of 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # 16 channels from second layer, map to 64 channels, 5x5 kernel
        self.conv3 = nn.conv2d(in_channels = 16, out_channels = 64, kernel_size = 5)

        # 64 channels from third layer, map to 128 channels, 5x5 kernel
        self.conv4 = nn.conv2d(in_channels = 64, out_channels = 128, kernel_size = 5)

        # 2x2 pooling kernel, stride of 2
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Fully connected layer
        self.fc1 = nn.Linear(in_features = 128*32*32, out_features = 10)
    def forward(self, x):
        pass

# %%
# Model Initialization
model = CNN()

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss() # Use Cross Entropy Loss; applies softmax
optimizer = torch.optim.Adam(model.parameters(), lr = eta) # Adam Optimizer

# %%
# Training loop
steps_per_epoch = len(train_loader)

# Loss Cache for plotting
loss_by_epoch = []

# Epoch iteration
for epoch in range(num_epochs):
    
    # refresh epoch loss at each epoch
    epoch_loss = 0

    # Batch Iteration
    for i, (x,y) in enumerate(train_loader):

        # refresh gradient
        optimizer.zero_grad()

        # forward pass
        y_hat = model(x)

        # evaluate loss
        loss = loss_fn(y_hat, y)

        # backprop
        loss.backward()

        # update
        optimizer.step()
        
        # add the batch loss to the epoch loss
        epoch_loss += loss.item()

        # progress tracker
        if (i+1)%1000 == 0:
            print(f'batch {i+1} in epoch {epoch + 1} complete')
    print(f'EPOCH {epoch + 1} TOTAL LOSS: {epoch_loss}')

    # Cache loss by epoch
    loss_by_epoch.append(epoch_loss)
# %%
# Test Performance

# plot epoch losses
plt.plot(range(1,num_epochs+1), loss_by_epoch)

with torch.no_grad():
    correct = 0
    total = 0
    class_correct = [0 for i in range(10)]
    class_totals = [0 for i in range(10)]
    total_correct = 0
    total_samples = 0

    for x, y in test_loader:

        # total accuracy 
        y_hat = model(x) # raw probs
        y_hat = torch.argmax(y_hat,dim=1) # choose highest prob class
        total_samples += y.size()[0]
        total_correct += (y_hat == y).sum().item()

        # classwise accuracy 
        for j in range(num_batches):
            label = y[j]
            predicted_class = y_hat[j]
            if predicted_class == label:
                class_correct[label] += 1
            class_totals[label] += 1
    
    # report total accuracy 
    total_acc = total_correct / total_samples
    print(f"Total accuracy: {total_acc}")

    # report class accuracy
    for i in range(10):
        cur_acc = class_correct[i]
        cur_total = class_totals[i]
        print(f"Class {classes[i]} accuracy: {cur_acc / cur_total}")

# %%
