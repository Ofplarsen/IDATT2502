import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm

x_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1,1]], dtype=torch.float).reshape(-1,2)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)

class NANDOperator:

    def __init__(self):
        self.W = torch.tensor(torch.rand(2, 1), requires_grad=True)
        self.b = torch.tensor(torch.rand(1, 1), requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def logits(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NANDOperator()

n = 10000000
lr = 0.0001
p = 100000
# Optimizer W, b, and learning rate
optimizer = torch.optim.SGD([model.W, model.b], lr)

for epoch in tqdm.tqdm(range(n)):
    model.loss(x_train, y_train).backward()  # Computes loss gradients
    if epoch % p == 0:
        print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))
    optimizer.step()  # Adjusts W and /or b
    optimizer.zero_grad() #Clears gradients for next step

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))


xt =x_train.t()[0]
yt =x_train.t()[1]

fig = plt.figure("Logistic regression: the logical NAND")

plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$\\hat y=f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.plot(xt.squeeze(), yt.squeeze(), y_train[:, 0].squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$", color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

table = plt.table(cellText=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
                  colWidths=[0.1] * 3,
                  colLabels=["$x_1$", "$x_2$", "$f(\\mathbf{x})$"],
                  cellLoc="center",
                  loc="lower right")


plot1_f.remove()
x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
y_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[(x1_grid[i, j]),  (x2_grid[i, j])]], dtype=torch.float))
plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")


fig.canvas.draw()

plt.show()