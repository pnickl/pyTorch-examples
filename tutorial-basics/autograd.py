import torch

# SIMPLE 1 LAYER NN
x = torch.ones(5)
y = torch.zeros(3)

w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

z = torch.matmul(x, w) + b

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print('Gradient function fr z =', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# COMPUTE BACKPROP DERIVATIVES
loss.backward()
# retrieve values
print(w.grad)
print(b.grad)

# STOP TRACKING GRADIENTS
z = torch.matmul(x, w) + b
print(z.requires_grad)

# method 1:
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

# method 2 with detach:
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

# use cases:
# - mark parameters as frozen: e.g. when finetuning a pretrained network
# - speed up computation during forward pass



# COMPUTE JACOBIAN-VECTOR PRODUCTS
# by calling backward with v as argument

inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
print(inp, out)

out.backward(torch.ones_like(inp), retain_graph=True)   # retain_graph for several calls
print("First call \n", inp.grad)
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nSecond call\n", inp.grad)
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print("\nCall after zeroing gradients\n", inp.grad)