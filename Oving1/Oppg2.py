import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressionModel3D:

    def __init__(self):
        self.W = torch.rand((2,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)



path = "resources/day_length_weight.csv"
data = pd.read_csv(path, dtype='float')

y_train = data.pop('day')
x_train = torch.tensor(data.to_numpy(), dtype=torch.float).reshape(-1, 2)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)

model = LinearRegressionModel3D()
n = 100000
lr = 0.0001
p = 1000
# Optimizer W, b, and learning rate
optimizer = torch.optim.SGD([model.W, model.b], lr)

for epoch in range(n):
    model.loss(x_train, y_train).backward()  # Computes loss gradients
    if epoch % p == 0:
        print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    optimizer.step()  # Adjusts W and /or b
    optimizer.zero_grad() #Clears gradients for next step

print("\nFinal: ")
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Plot

xt =x_train.t()[0]
yt =x_train.t()[1]

fig = plt.figure('Linear regression 3d')
ax = fig.add_subplot(projection='3d', title="Model for predicting days lived by weight and length")
# Plot
ax.scatter(xt.numpy(),  yt.numpy(), y_train.numpy(),label='$(x^{(i)},y^{(i)}, z^{(i)})$')
ax.scatter(xt.numpy(),yt.numpy() ,model.f(x_train).detach().numpy() , label='$\\hat y = f(x) = xW+b$', color="orange")
ax.legend()
plt.show()
