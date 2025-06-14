import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the domain
x = np.linspace(-2, 2, 1000)

# True periodic x^2 function
def true_x2_periodic(x):
    x_mod = ((x + 1) % 2) - 1
    return x_mod ** 2

# Fourier approximation function
def fourier_x2(x, n_terms):
    a0 = 1 / 3
    result = a0 * np.ones_like(x)
    for n in range(1, n_terms + 1):
        an = (4 * (-1)**n) / (n**2 * np.pi**2)
        result += an * np.cos(n * np.pi * x)
    return result

# Black background style setup
plt.style.use('dark_background')  # 🎩✨ Night mode activated

# Plot setup
fig, ax = plt.subplots()
fig.patch.set_facecolor('black')  # outer figure background
ax.set_facecolor('black')         # plot background
ax.set_xlim(-2, 2)
ax.set_ylim(-0.2, 1.2)

# Plot lines
line, = ax.plot([], [], 'r', lw=2, label='Fourier Approximation')  # red line
true_line, = ax.plot(x, true_x2_periodic(x), color='cyan', linestyle='--', alpha=0.5, label=r'True $x^2$ Periodic')

# Grid & legend
ax.grid(True, color='gray', linestyle=':', alpha=0.4)
ax.legend(facecolor='black', edgecolor='white', loc='upper right')
title = ax.set_title("", color='white')

# Animation update function
def update(n):
    y = fourier_x2(x, n)
    line.set_data(x, y)
    title.set_text(f"Fourier Series of $x^2$ with {n} Terms")
    return line, title

# Create animation
ani = FuncAnimation(fig, update, frames=range(1, 51), interval=100, blit=True)

plt.show()
