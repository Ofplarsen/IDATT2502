import torch
import matplotlib.pyplot as plt
import pandas as pd


class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


def oppg3():
    path = "resources/day_head_circumference.csv"
    data = pd.read_csv(path, delimiter=',', header=None, names=['length', 'weight'])

    data['length'].pop(0)
    data['weight'].pop(0)

    x_data = data['length'].values.astype(float).tolist()
    y_data = data['weight'].values.astype(float).tolist()

    print(x_data)
    print(y_data)

    x_train = torch.tensor(x_data).reshape(-1, 1)
    y_train = torch.tensor(y_data).reshape(-1, 1)

    model = LinearRegressionModel()
    n = 1000000
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

    plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel('Length')
    plt.ylabel('Weight')
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
    plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.gcf().text(0.15, 0.93,"n: " + str(n) + ", lr: " + str(lr) + ", loss: " + str(model.loss(x_train,y_train)))
    plt.legend()
    plt.show()
