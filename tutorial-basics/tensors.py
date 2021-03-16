import torch
import numpy as np

# INITIALIZING A TENSOR

# from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# from numpy array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# from another tensor
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# with random or constant values
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# TENSOR ATTRIBUTES

tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# OPERATIONS ON TENSORS

# move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# indexing, slicing
tensor = torch.ones(4, 4)
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# concatenate tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# ARITHMETIC OPERATIONS

# matmul
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

# element-wise mul
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# single-element tensors
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item), agg)

# in-place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)


# BRIDGE WITH NUMPY

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")