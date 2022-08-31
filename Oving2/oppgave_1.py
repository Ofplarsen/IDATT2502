import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

x_train = torch.tensor([[0.0], [1.0]], dtype=torch.float).reshape(-1, 1)
y_train = torch.tensor([[1], [0]], dtype=torch.float)


class NotOperator:

    def __init__(self):
        self.W = torch.tensor(torch.rand(1, 1), requires_grad=True)
        self.b = torch.tensor(torch.rand(1, 1), requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


def program():
    model = NotOperator()

    n = 100000
    lr = 0.1
    p = 100000
    # Optimizer W, b, and learning rate
    optimizer = torch.optim.SGD([model.W, model.b], lr)

    for epoch in tqdm.tqdm(range(n)):
        model.loss(x_train, y_train).backward()  # Computes loss gradients
        if epoch % p == 0:
            print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))
        optimizer.step()  # Adjusts W and /or b
        optimizer.zero_grad()  # Clears gradients for next step

    print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

    plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
    x = torch.arange(0.0, 1.0, 0.01).reshape(-1, 1)
    plt.plot(x, model.f(x).detach().numpy(), label='y = f(x) = 1 / (1 + np.exp(-z))')
    plt.gcf().text(0.15, 0.93,
                   "n: " + str(n) + ", lr: " + str(lr) + ", loss: " + str(model.loss(x_train, y_train).data))
    plt.legend()
    plt.show()
