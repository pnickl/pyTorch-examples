import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

z = y * y * 3
out = z.mean()

print(z, out)

# .required_grad(...) changes existing Tensor's requires_grad flag in-place. The input flag
# defaults to False if not given
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

# backprop on scalar
out.backward()
print(x.grad)

# backprop on vector
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
print(y)

# stop tracking histories on tensors: method 1, torch.no_grad
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# stop tracking histories on tensors: method 2, detach