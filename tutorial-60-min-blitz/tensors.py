import torch
import numpy as np

# INTIALIZE
# init from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# init from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# init from other tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# init with random or constant value
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# TENSOR ATTRIBUTES
tensor = torch.rand(3, 4)
print(tensor.shape, tensor.dtype, tensor.device)

# TENSOR OPERATIONS
# move tensor to gpu if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Standard numpy indexing / slicing
tensor = torch.ones(4, 4)
tensor[:, 1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# element-wise products
print(tensor.mul(tensor))
print(tensor * tensor)

# matrix product
print(tensor.matmul(tensor.T))
print(tensor @ tensor.T)

# in-place operationrs
tensor.add_(5)

# BRIDGE WITH NUMPY
# tensor to numpy array
t = torch.ones(5)
n = t.numpy()

# change in tensor reflects in numpy array
t.add_(1)
print(t, n)

# numpy to tensor
n = np.ones(5)
t = torch.from_numpy(n)

# changes in numpy reflect in tensor
np.add(n, 1, out=n)
print(t, n)

