import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the domain
x = np.linspace(-2, 2, 1000)  # shows 2 periods

# True periodic x^2 function (periodic extension)
def true_x2_periodic(x):
    x_mod = ((x + 1) % 2) - 1  # map x to [-1, 1]
    return x_mod ** 2

# Fourier coefficients for x^2 on [-1, 1]
def fourier_x2(x, n_terms):
    a0 = 1 / 3
    result = a0 * np.ones_like(x)
    for n in range(1, n_terms + 1):
        an = (4 * (-1)**n) / (n**2 * np.pi**2)
        result += an * np.cos(n * np.pi * x)
    return result

# Plot setup
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-0.2, 1.2)
line, = ax.plot([], [], 'r', label='Fourier Approximation')
true_line, = ax.plot(x, true_x2_periodic(x), 'k--', alpha=0.4, label=r'True $x^2$ Periodic')
ax.legend()
ax.grid(True)
ax.set_title("Fourier Series of Periodic $x^2$ on [-1, 1]")

# Animation function
def update(n):
    y = fourier_x2(x, n)
    line.set_data(x, y)
    ax.set_title(f"Fourier Series with {n} Terms")
    return line,

# Animate
ani = FuncAnimation(fig, update, frames=range(1, 51), interval=100, blit=True)
plt.show()
