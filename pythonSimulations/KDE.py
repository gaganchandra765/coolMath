import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Step 1: Generate 500 samples from a mixture of Gaussians
n = 500
# Mixture: 0.5 * N(-2, 1) + 0.5 * N(2, 0.5)
samples = np.concatenate([
    np.random.normal(-2, 1, int(n / 2)),  # 0.5 * N(-2, 1)
    np.random.normal(2, 0.5, int(n / 2))   # 0.5 * N(2, 0.5)
])

# Step 2: Define the Gaussian kernel function
def gaussian_kernel(u):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

# Step 3: Implement KDE function
def kde(samples, x, h):
    n = len(samples)
    kernels = np.sum(gaussian_kernel((x - samples) / h))
    return kernels / (n * h)

# Step 4: Create a fine grid of x values for plotting
x_grid = np.linspace(-6, 6, 1000)

# Step 5: Plot KDE for different bandwidths
bandwidths = [0.1, 0.4, 1.0]
plt.figure(figsize=(10, 6))

# Plot the true density of the mixture
true_density = 0.5 * norm.pdf(x_grid, -2, 1) + 0.5 * norm.pdf(x_grid, 2, 0.5)
plt.plot(x_grid, true_density, label="True Density", color="black", linewidth=2)

# Plot KDE for different bandwidths
for h in bandwidths:
    kde_values = [kde(samples, x, h) for x in x_grid]
    plt.plot(x_grid, kde_values, label=f'KDE (h={h})', linewidth=2)

# Step 6: Finalizing the plot
plt.title("Kernel Density Estimation for a Mixture of Gaussians")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
