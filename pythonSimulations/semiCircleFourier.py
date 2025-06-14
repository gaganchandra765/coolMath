import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the original semicircle function on [-1, 1]
def semicircle(x):
    return np.sqrt(1 - x**2) if -1 <= x <= 1 else 0

# Define the period
L = 1  # Half-period (so full period is 2L = 2)

# Fourier coefficients
def a0():
    integral, _ = quad(lambda x: semicircle(x), -L, L)
    return (1 / L) * integral

def an(n):
    integral, _ = quad(lambda x: semicircle(x) * np.cos(n * np.pi * x / L), -L, L)
    return (1 / L) * integral

def bn(n):
    integral, _ = quad(lambda x: semicircle(x) * np.sin(n * np.pi * x / L), -L, L)
    return (1 / L) * integral

# Reconstruct the Fourier series
def fourier_series(x, N):
    sum = a0() / 2
    for n in range(1, N+1):
        sum += an(n) * np.cos(n * np.pi * x / L) + bn(n) * np.sin(n * np.pi * x / L)
    return sum

# Plotting
x_vals = np.linspace(-3, 3, 1000)
N_terms = [1, 5, 10, 50]  # Try with different numbers of terms

plt.figure(figsize=(12, 8))
for i, N in enumerate(N_terms):
    y_vals = [fourier_series(x, N) for x in x_vals]
    plt.plot(x_vals, y_vals, label=f'N = {N}')

# Plot the original repeated semicircle just for comparison
x_orig = np.linspace(-1, 1, 500)
y_orig = [semicircle(x) for x in x_orig]
for shift in range(-2, 3):
    plt.plot(x_orig + 2 * shift, y_orig, 'k--', alpha=0.3)

plt.title("Fourier Series Approximation of a Repeating Semicircle")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)
plt.show()
