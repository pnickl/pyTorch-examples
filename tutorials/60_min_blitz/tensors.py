# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

from __future__ import print_function
import torch

# initializing
x = torch.empty(5, 3)
y = torch.rand(5, 3)
z = torch.zeros(5, 3, dtype=torch.long)
a = torch.tensor([5.4, 3])

x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
                                              # result has the same size

# adding
y = torch.rand(5, 3)

y = torch.rand(5, 3)

result = torch.empty(5, 3)
torch.add(x, y, out=result)

y.add_(x)

# indexing
print(x[:, 1])

# resizing
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# get value as python number
x = torch.randn(1)
print(x)
print(x.item())

# torch tensor to numpy array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# numpy array to torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# move tensors into device
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

