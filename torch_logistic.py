
# PyTorch Example: LOGISTIC REGRESSION

#%%
# Imports
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
# Data Loading
cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target
n, p = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Data Processing
sc = StandardScaler() # standardizes data
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

n, p = X_train.shape # new shape
y_train = y_train.view(n, 1)
y_test = y_test.view(y_test.shape[0], 1)

# %%
# Model initialization via class inheritance
class LogisticRegression(nn.Module):

    # set to inherit the init of a general neural net
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        
        # initialize a linear layer
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, X):
        y_hat = torch.sigmoid(self.linear(X))
        return y_hat

# Model Instance Creation
model = LogisticRegression(p)
# %%
# Loss and Optimizer
loss_func = nn.BCELoss() #Binary Cross Entropy Loss
eta = .01
optimizer = torch.optim.SGD(model.parameters(), lr = eta)
# %%
# Training loop
num_epochs = 100
for epoch in range(num_epochs):

    # forward pass and loss
    y_hat = model(X_train)
    loss = loss_func(y_hat, y_train)

    # backward pass
    loss.backward()

    # weight update
    optimizer.step()

    # grad refresh
    optimizer.zero_grad

    # progress update
    if (epoch+1) % 10 == 0:
        print("Epoch", epoch+1, "out of", num_epochs, "with current loss", loss.item())

# disables gradient tracking for speed of forward passes
with torch.no_grad():
    y_hat = model(X_test)
    y_hat_class = y_hat.round() # round to binary class, sigmoid in (0,1)
    acc = y_hat_class.eq(y_test).sum() / float(y_test.shape[0])
    print("Accuracy on Testing Data:", acc)
# %%
