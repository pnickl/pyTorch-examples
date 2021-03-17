import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass
prediction = model(data)

# calculate loss and backprop (stores gradients in parameter's .grad attribute)
loss = (prediction - labels).sum()
loss.backward()

# load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# gradient descent
optim.step()

# DETAILS: DIFFERENTIATION IN AUTOGRAD
a = torch.tensor([2, 3], requires_grad=True)
b = torch.tensor([6, 4], requires_grad=True)

Q = 3*a**3 - b**2

# calculate gradient
external_grad = torch.tensor([1, 1])
# loss function is not scalar-valued, we thus need to pass a vector to backward()
# backward calculates jacobian-vector product
Q.backward(gradient=external_grad)

# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# DETAILS: MANIPULATE THE COMPUTATIONAL GRAPH
x = torch.rand(5, 5)        # frozen params
y = torch.rand(5, 5)        # frozen params
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

from torch import nn, optim
model = torchvision.models.resnet18(pretrained=True)

# freeze all params
for param in model.parameters():
    param.requires_grad = False

# finetune model on new dataset / unfreeze last layer
model.fc = nn.Linear(512, 10)       # this is the classifier; the last linear layer; replace with unfrozen layer

# optimize only the classifer
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

# alternative way of freezing parameters
# with torch.no_grad()