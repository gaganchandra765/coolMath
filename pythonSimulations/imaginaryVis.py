import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Time domain
t = np.linspace(-10, 10, 1000)
f = np.exp(-t**2) * np.cos(5*t)

# Fourier transform
F = np.fft.fftshift(np.fft.fft(f))
freq = np.fft.fftshift(np.fft.fftfreq(len(t), d=(t[1]-t[0])))

# Extract real and imaginary parts
real_part = np.real(F)
imag_part = np.imag(F)

# Plot 3D
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(freq, real_part, imag_part, color='C3', lw=2)
ax.set_title("3D Visualization of Fourier Transform (Real vs Imag vs Frequency)", pad=20)
ax.set_xlabel("Frequency ω")
ax.set_ylabel("Re[F(ω)]")
ax.set_zlabel("Im[F(ω)]")

# Optional: nice viewing angle
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()

