# https://pytorch.org/tutorials/beginner/nn_tutorial.html

# download MNIST data
from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# load data and inspect a sample
import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

# concert data to torch.tensor
import torch

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
print(x_train, x_train.shape, y_train.min(), y_train.max())

# neural net from scratch
import math
weights = torch.randn(784, 10) / math.sqrt(784) #this ix Xavier init - mutliply by 1/sqrt(n)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x): # custom activation
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

# forward pass
bs = 64

xb = x_train[0:bs]
preds = model(xb)
print(preds[0], preds.shape)

# define loss
def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))

# calc accuracy of prediction
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))

#from IPython.core.debugger import set_trace

lr = 0.5    # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

print(loss_func(model(xb), yb))

# REFACTOR USING FUNCTIONAL
import torch.nn.functional as F

loss_func = F.cross_entropy()
def model(xb):
    return xb @ weights + bias

# REFACTOR USING DATASET
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
xb, yb = train_ds[i*bs : i*bs+bs]

# REFACTOR USING nn.MODULE
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

        def forward(self, xb):
            return xb @ self.weights + self.bias

model = Mnist_Logistic()
print(loss_func(model(xb), yb))



def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i * bs
            xb, yb = train_ds[i * bs: i * bs + bs]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): p -= p.grad * lr
                model.zero_grad()


# REFACTOR USING NN.LINEAR
# REFACTOR USING OPTIM
# REFACTOR USING DATASET
# REFACTOR USING DATALOADER