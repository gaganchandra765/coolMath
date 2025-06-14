import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Basic settings
fig, ax = plt.subplots()
x = np.linspace(-np.pi, np.pi, 1000)
line, = ax.plot([], [], 'r', label='Fourier Approximation')
true_square, = ax.plot(x, np.sign(np.sin(x)), 'k--', alpha=0.3, label='True Square Wave')

# Formatting
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title("Fourier Series Approximation of a Square Wave")
ax.legend()
ax.grid(True)

# Fourier series approximation of a square wave
def fourier_square_wave(x, n_terms):
    result = np.zeros_like(x)
    for n in range(1, n_terms + 1):
        k = 2*n - 1  # Only odd harmonics
        result += (1 / k) * np.sin(k * x)
    return (4 / np.pi) * result

# Animation update function
def update(n):
    y = fourier_square_wave(x, n)
    line.set_data(x, y)
    ax.set_title(f"Fourier Series Approximation with {2*n - 1} Harmonics")
    return line,

# Create the animation
ani = FuncAnimation(fig, update, frames=range(1, 51), interval=100, blit=True)

# Show the animation
plt.show()
