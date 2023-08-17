# PyTorch Example: LINEAR REGRESSION
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Generation of synthetic data
X_np, y_np = datasets.make_regression(n_samples = 100, n_features = 1, noise = 10)
N = y_np.shape[0]

# Port data to PyTorch and reshape
X = torch.from_numpy(X_np.astype(np.float32))
y = torch.from_numpy(y_np.astype(np.float32))
y = y.view(N, 1)
n, p = X.shape

# Model construction
model = nn.Linear(p, 1)

# Loss and Optimizer construction
loss = nn.MSELoss()
eta = .01
optimizer = torch.optim.SGD(model.parameters(), lr = eta)

# Training
epoch_count = 100
for epoch in range(epoch_count):

    # forward pass and loss
    y_hat = model(X)
    loss_val = loss(y_hat, y)

    # backprop; stores gradients so step can be called
    loss_val.backward()

    # update parameters
    optimizer.step()

    # gradient reset
    optimizer.zero_grad()

    # progress report
    if (epoch + 1) % 10 == 0:
        print('epoch', epoch + 1, 'complete. Loss is', loss_val.item())

# plotting; for quantities to be plotted, detatch from comp tree
y_hat = model(X).detach().numpy()
plt.plot(X_np, y_np, 'ro')
plt.plot(X_np, y_hat, 'b')
plt.show()
