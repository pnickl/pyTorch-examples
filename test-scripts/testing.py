import torch
import numpy as np
import pandas as pd

a = torch.tensor([i for i in range(5)])

a[3]
a[3].item()
a_col = a.view([5,1])

np_array = np.arange(1.0,3.5,0.5)
tensor_from_array = torch.from_numpy(np_array)
back_to_np = tensor_from_array.numpy()

pd_series = pd.Series([i for i in range(1,6)])
pd_to_torch = torch.from_numpy(pd_series.values)
tp_list = pd_to_torch.tolist()

list = []

for i in range(3):
    inner_list = []
    for j in range(3):
        inner_list.append(10*(i+1)+j)
    list.append(inner_list)

A = torch.tensor(list)
A.transpose(0, 1)
A.shape

B = torch.tensor(np.arange(1, 7, 1))
B.view(2,3)

# autograd example 1
def func(x):
    return (x**3 - 7*x**2 + 11*x)
x = torch.tensor(2.0, requires_grad=True)
y = func(x)
y.backward()
print(x.grad)

# autograd example 2
u = torch.tensor(2.0, requires_grad=True)
v = torch.tensor(1.0, requires_grad=True)
f = 3*u**2*v - 4*v**3
f.backward(retain_graph=True)
print(u.grad, v.grad)

# autograd example 3
x = torch.linspace(-10, 10, requires_grad=True)
x_squared = x**2
y = torch.sum(x**2)
y.backward()
print(x.grad)

from torch import nn
class Network(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super(Network, self).__init__()

        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.hidden2 = nn.Linear(n_hidden1, n_hidden2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(n_hidden2, n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# create data
from sklearn.datasets import make_classification,make_regression
import numpy as np
import matplotlib.pyplot as plt

input_dim = 5
data_points = 100

X, y = make_classification(data_points, input_dim, n_informative=3, random_state=101)
X = X.astype(np.float32)
y = y.astype(np.float32)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

n_input = X.shape[1] # Must match the shape of the input features
n_hidden1 = 8 # Number of neurons in the 1st hidden layer
n_hidden2 = 4 # Number of neurons in the 2nd hidden layer
n_output = 1 # Number of output units (for example 1 for binary classification)
model = Network(n_input, n_hidden1, n_hidden2, n_output)
print(model)

criterion = nn.BCELoss()

logits = model.forward(X)

logits_numpy = model.forward(X).detach().numpy().flatten()
plt.figure(figsize=(15,3))
plt.title("Output probabilities with the untrained model",fontsize=16)
plt.bar([i for i in range(100)],height=logits_numpy)
plt.show()

loss = criterion(logits, y)
print(loss.item())

print("Gradients of the weights of the 2nd hidden layer connections before computing gradient:\n",model.hidden2.weight.grad)
loss.backward() # Compute gradients
print("Gradients of the weights of the 2nd hidden layer connections after computing gradient:\n",model.hidden2.weight.grad)

from torch import optim
optimizer = optim.SGD(model.parameters(),lr=0.1)

epochs = 1000
running_loss = []

for i, e in enumerate(range(epochs)):
    optimizer.zero_grad()
    output = model.forward(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    running_loss.append(loss.item())

plt.figure(figsize=(7,4))
plt.title("Loss over epochs",fontsize=16)
plt.plot([e for e in range(epochs)],running_loss)
plt.grid(True)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Training loss",fontsize=15)
plt.show()