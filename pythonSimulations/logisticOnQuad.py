import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 1. Generate the Parabola Dataset 📐
np.random.seed(42)
num_points = 1000
X = np.random.uniform(-2, 2, (num_points, 1))
Y = np.random.uniform(-1, 4, (num_points, 1))  # y range wider to get above/below

noise = 0.1 * np.random.randn(num_points, 1)
labels = (Y > (X**2 + noise)).astype(int)  # Class 1 if above the noisy parabola

data = np.hstack((X, Y))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 2. Neural Network Class
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

net = NeuralNet()

# 3. Training setup
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 4. Train the Neural Network
epochs = 1000
for epoch in range(epochs):
    net.train()
    preds = net(X_train_tensor)
    loss = criterion(preds, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# 5. Evaluate
net.eval()
with torch.no_grad():
    acc = ((net(X_test_tensor) > 0.5) == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {acc.item()*100:.2f}%")

# 6. Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:,0].min() - 0.2, X[:,0].max() + 0.2
    y_min, y_max = X[:,1].min() - 0.2, X[:,1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model(grid).reshape(xx.shape)
        
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, probs, levels=50, cmap='coolwarm', alpha=0.8)
    plt.scatter(X[:,0], X[:,1], c=y.squeeze(), cmap='bwr', edgecolors='k', s=20)
    plt.title("Decision Boundary: Points Above vs Below y = x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

plot_decision_boundary(net, data, labels)
