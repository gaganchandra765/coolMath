import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set up the figure
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_title(r"$e^{ix} = \cos x + i \sin x$ on the Argand Plane", fontsize=14)
ax.set_xlabel("Real")
ax.set_ylabel("Imaginary")

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 1000)
circle, = ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linestyle='--')

# Point for e^{ix}
point, = ax.plot([], [], 'ro', label=r"$e^{ix}$")
cos_line, = ax.plot([], [], 'b-', label=r"Re: $\cos x$")
sin_line, = ax.plot([], [], 'g-', label=r"Im: $\sin x$")

# Origin
ax.plot(0, 0, 'ko')  # origin
ax.legend(loc='upper right')

# Animation update function
def update(frame):
    x = frame
    z = np.exp(1j * x)
    
    # Update point
    point.set_data([z.real], [z.imag])
    
    # Update lines
    cos_line.set_data([0, z.real], [0, 0])
    sin_line.set_data([z.real, z.real], [0, z.imag])
    
    return point, cos_line, sin_line

# Create animation
x_vals = np.linspace(0, 2*np.pi, 200)
ani = animation.FuncAnimation(fig, update, frames=x_vals, interval=50, blit=True)

plt.grid(True)
# Show the animation
plt.show()

