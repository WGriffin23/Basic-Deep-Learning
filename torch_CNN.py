# PyTorch Example: CONVOLUTIONAL NEURAL NETWORK
'''
Remarks:
- CIFAR-10 is a famous 10 class machine learning dataset
- Data are 32x32 color images 
- There are 6000 images per class, for a total of 60,000 images
- Training and Testing datasets are pre-curated at a 5:1 ratio
- TEST COMMENT
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
num_epochs = 20
num_batches = 4
eta = .003 # learning rate

# %%
# Data Loading

# We want to standardize channel intensities and work with a Tensor
transform = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))])

# Loaders; torchvision already caches these as Dataset objects
train = torchvision.datasets.CIFAR10(root = '.', train = True,
download = True, transform = transform)

test = torchvision.datasets.CIFAR10(root = '.', train = False,
download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train, batch_size = num_batches,
shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = num_batches,
shuffle = False)

# Manually cache classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
'ship', 'truck')

# %%
# Model Design
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6,
        kernel_size = 5)

        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # need in channels from first conv layer
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, 
        kernel_size = 5)

        self.fc1 = nn.Linear(in_features= 16*10*10, out_features = 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = out.view(-1, 16*10*10) # flatten before linear
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # no softmax, we are using CrossEntropyLoss()

        return out

# %%
# Model Initialization
model = CNN()

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = eta)

# %%
# Training loop
steps_per_epoch = len(train_loader)

# loss follower
loss_by_epoch = []

for epoch in range(num_epochs):
    epoch_loss = 0
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
