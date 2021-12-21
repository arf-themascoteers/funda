import torch
import torch.nn as nn
import matplotlib.pyplot as plt

linear = nn.Sequential(
    nn.Linear(1,2),
    nn.ReLU()
)

linear = torch.load("linear.h5")

x = torch.linspace(-5,5, steps=11).reshape(-1, 1)
y = linear(x)
z = y.sum(axis=1)

plt.plot(x.squeeze().detach().numpy(), y.squeeze().detach().numpy())
plt.show()
plt.plot(x.squeeze().detach().numpy(), z.squeeze().detach().numpy())
plt.show()
y[0][1] = 2*y[0][1]
plt.plot(x.squeeze().detach().numpy(), y.squeeze().detach().numpy())
plt.show()
plt.plot(x.squeeze().detach().numpy(), z.squeeze().detach().numpy())
plt.show()


