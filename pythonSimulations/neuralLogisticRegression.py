import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Generate Dataset (Donut Data 🍩)
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Define Neural Network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # For binary classification (output between 0 and 1)
        )

    def forward(self, x):
        return self.model(x)

net = NeuralNet()

# 3. Training Setup
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
epochs = 1000

# 4. Train the Network
for epoch in range(epochs):
    net.train()
    y_pred = net(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. Evaluate Accuracy (optional)
with torch.no_grad():
    net.eval()
    preds = net(X_test_tensor)
    acc = ((preds > 0.5).float() == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {acc.item()*100:.2f}%")

# 6. Plot the Decision Boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
    y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)

    plt.figure(figsize=(7,7))
    plt.contourf(xx, yy, probs, levels=50, cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolor='k')
    plt.title("Non-linear Decision Boundary via Neural Network")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

# Show the final result
plot_decision_boundary(net, X, y)
