import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd


class NotLinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor(torch.rand(1, 1), requires_grad=True)
        self.b = torch.tensor(torch.rand(1, 1), requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

    def sigmoid(self, z): #Didn't work
        return 1 / (1 + np.exp(-z))



path = "resources/day_head_circumference.csv"
data = pd.read_csv(path, dtype='float')

y_train = data.pop('head circumference')
x_train = torch.tensor(data.to_numpy(), dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float).reshape(-1, 1)


model = NotLinearRegressionModel()
n = 1000000
lr = 0.000001
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

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Day')
plt.ylabel('head circumference')
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='y = f(x) = 1 / (1 + np.exp(-z))')
plt.gcf().text(0.15, 0.93,"n: " + str(n) + ", lr: " + str(lr) + ", loss: " + str(model.loss(x_train,y_train).data))
plt.legend()
plt.show()
