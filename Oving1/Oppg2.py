import torch
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class LinearRegressionModel3D:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


def oppg2():
    path = "resources/day_length_weight.csv"
    data = pd.read_csv(path, delimiter=',', header=None, names=['day', 'length', 'weight'])

    data['day'].pop(0)
    data['length'].pop(0)
    data['weight'].pop(0)

    x_data = data['day'].values.astype(float).tolist()
    y_data = data['length'].values.astype(float).tolist()
    z_data = data['weight'].values.astype(float).tolist()



    print(x_data)
    print(y_data)
    print(z_data)

    x_train = torch.tensor(x_data).reshape(-1, 1)
    y_train = torch.tensor(y_data).reshape(-1, 1)
    z_train = torch.tensor(z_data).reshape(-1, 1)

    model1 = LinearRegressionModel3D()
    model2 = LinearRegressionModel3D()
    n = 1000000
    lr = 0.0000001
    p = 10000
    # Optimizer W, b, and learning rate
    optimizer1 = torch.optim.SGD([model1.W, model1.b], lr)
    optimizer2 = torch.optim.SGD([model2.W, model2.b], lr)

    for epoch in range(n):
        model1.loss(x_train, y_train).backward()  # Computes loss gradients
        if epoch % p == 0:
            print(str(int (epoch/p)) + " of "+ (str(int(n/p))) + "1: W = %s, b = %s, loss = %s" % (model1.W, model1.b, model1.loss(x_train, y_train)))
        optimizer1.step()  # Adjusts W and /or b
        optimizer1.zero_grad() #Clears gradients for next step

    for epoch in range(10000):
        model2.loss(x_train, z_train).backward()  # Computes loss gradients
        if epoch % p == 0:
            print("2: W = %s, b = %s, loss = %s" % (model2.W, model2.b, model2.loss(x_train, z_train)))
        optimizer2.step()  # Adjusts W and /or b
        optimizer2.zero_grad()  # Clears gradients for next step



    print("\nFinal: ")
    print("1: W = %s, b = %s, loss = %s" % (model1.W, model1.b, model1.loss(x_train, y_train)))
    print("2: W = %s, b = %s, loss = %s" % (model2.W, model2.b, model2.loss(x_train, z_train)))

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax = fig.add_subplot(projection='3d')

    ax.scatter(xs=x_train, ys=y_train, zs=z_train)

    ax.set_title("Age-wise body weight-height distribution")

    ax.set_xlabel("Day ")

    ax.set_ylabel("Length")

    ax.set_zlabel("Weight")
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
    ax.plot_wireframe(x, model1.f(x).detach(), model2.f(x).detach())
    plt.show()
