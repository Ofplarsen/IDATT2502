import torch
import numpy as np


class LinearRegressionModel3D:

    def __init__(self):
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.tensor(-0.4, requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


x = torch.tensor([[0.4], [0.4]], dtype=torch.float).reshape(-1, 2)

model = LinearRegressionModel3D()

print(model.f(x))
