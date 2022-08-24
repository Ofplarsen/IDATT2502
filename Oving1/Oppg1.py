import torch
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    #Predictor
    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

def oppg1():
    path = "resources/length_weight.csv"
    data = pd.read_csv(path, delimiter=',', header=None, names=['length', 'weight'])

    data['length'].pop(0)
    data['weight'].pop(0)
    print(data['length'].values.tolist())


    x_data = data['length'].values.astype(float).tolist()
    y_data = data['weight'].values.astype(float).tolist()

    x_train = torch.tensor(x_data).reshape(-1, 1)
    y_train = torch.tensor(y_data).reshape(-1, 1)

    model = LinearRegressionModel()

    # Optimizer W, b, and learning rate
    optimizer = torch.optim.SGD([model.W, model.b], 0.01)

    for epoch in range(1000):
        model.loss(x_train, y_train).backward()  # Computes loss gradients
        if (epoch % 20 == 0):
            print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
        optimizer.step()  # Adjusts W and /or b

    print("\nFinal: ")
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

    # Plot

    plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
    plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
    plt.legend()
    plt.show()