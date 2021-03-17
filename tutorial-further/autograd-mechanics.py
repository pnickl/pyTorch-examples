# https://pytorch.org/docs/stable/notes/autograd.html

import torch, torchvision
import torch.nn as nn
from torch import optim

# excluding subgraphs from backward

x = torch.randn(5, 5)
y = torch.randn(5, 5)
z = torch.randn((5, 5), requires_grad=True)
a = x + y
a.requires_grad
b = a + z
b.requires_grad


model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
model.fc = nn.Linear(512, 100)

# Optimize only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# Multithreaded autograd

# Define a train function to be used in different threads
def train_fn():
    x = torch.ones(5, 5, requires_grad=True)
    # forwardd
    y = (x + 3) * (x + 4) * 0.5
    # backward
    y.sum().backward()
    # potenzial optimier uodate

# User write their own threading code to drive the train_fn
threads = []
for _ in range(10):
    p = threading.Thread(target=train_fn, args=())
    p.start()
