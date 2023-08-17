# PyTorch Example: ARTIFICIAL NEURAL NETWORK

#%%
# Imports
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# %%
# Parameter Initializations
p = 784 # images are 28x28 and vectorized
m1 = 100 
class_count = 10
num_epochs = 2
batch_size = 100
eta = .001

#%%

# Data Loading

# send to current dir, use training partition, bring in as tensor, 
# and download training data to current dir
train = torchvision.datasets.MNIST(root = '.', train = True,
transform = transforms.ToTensor(), download = True)

# send to current dir, use testing partition, bring in as tensor,
# and DO NOT download testing data
test = torchvision.datasets.MNIST(root = '.', train = False,
transform = transforms.ToTensor(), download = False)

# Training data into iterable DataLoader, which helps with batches
train_loader = torch.utils.data.DataLoader(dataset = train, 
batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test, 
batch_size = batch_size, shuffle = False)

# %%
# Neural Network Class
class ANN(nn.Module):

    def __init__(self, p, m1, C):
        '''
        p: input size
        m1: hidden layer 1 size
        C: number of classes
        '''
        # inherit the NeuralNet constructor, which does not require args
        super(ANN, self).__init__()

        # customize by adding our own layers using nn layer objects
        self.l1 = nn.Linear(p, m1) # linear part, no activation
        self.relu = nn.ReLU() # activation function
        self.l2 = nn.Linear(m1, C) # output layer, C class scores

    def forward(self, x):
        '''
        -Optimizers need to find the forward method, which we customize
        -Apply the layers you built in the custom inherited constructor
        -Build to process a single input passed as arg x
        '''
        out = self.l1(x) 
        out = self.relu(out)
        out = self.l2(out)
        return out

# %%
# TRAINING BLOCK 

# Model Initialization
model = ANN(p, m1, class_count)

# loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = eta)

# Training Loop
num_steps = len(train_loader)
for epoch in range(num_epochs):

    # batch loader for training data
    for i, (images, labels) in enumerate(train_loader):
        # images by default are 100x1x28x28 (num, channels, n, n)
        images = images.reshape(-1, 28*28)

        # forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # backward pass
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("epoch", epoch + 1, "out of", num_epochs, "step", 
            i+1, "out of", num_steps, "loss is", loss.item())


#%%

# Testing Loop
with torch.no_grad(): # deactivate gradient comps for speed

    n_right = 0
    n_samples = 0

    # iterate over batch
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        y_hat = model(images) # raw probabilities
        y_hat = torch.argmax(y_hat, dim = 1)

        # log accuracy
        n_samples += labels.shape[0]
        n_right += (y_hat == labels).sum().item() # this is a tensor

    # total accuracy
    acc = (n_right / n_samples)
    print("Testing Accuracy:", acc)


# %%
