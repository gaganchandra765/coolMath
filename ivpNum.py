import numpy as np
import matplotlib.pyplot as plt

# Define the differential equation
def f(x, y):
    return x + y

# Exact solution (for comparison)
def exact_solution(x):
    return 2 * np.exp(x) - x - 1

# Euler method
def euler(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    for i in range(n):
        y_new = y[-1] + h * f(x[-1], y[-1])
        x.append(x[-1] + h)
        y.append(y_new)
    return x, y

# Modified Euler (Heun's method)
def modified_euler(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    for i in range(n):
        xi, yi = x[-1], y[-1]
        k1 = f(xi, yi)
        k2 = f(xi + h, yi + h * k1)
        y_new = yi + (h / 2) * (k1 + k2)
        x.append(xi + h)
        y.append(y_new)
    return x, y

# Runge-Kutta 4
def rk4(f, x0, y0, h, n):
    x = [x0]
    y = [y0]
    for i in range(n):
        xi, yi = x[-1], y[-1]
        k1 = f(xi, yi)
        k2 = f(xi + h/2, yi + h/2 * k1)
        k3 = f(xi + h/2, yi + h/2 * k2)
        k4 = f(xi + h, yi + h * k3)
        y_new = yi + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x.append(xi + h)
        y.append(y_new)
    return x, y

# Parameters
x0, y0 = 0, 1
h = 0.2
n = int((2 - x0) / h)

# Compute solutions
x_e, y_e = euler(f, x0, y0, h, n)
x_me, y_me = modified_euler(f, x0, y0, h, n)
x_rk, y_rk = rk4(f, x0, y0, h, n)
x_vals = np.linspace(0, 2, 200)
y_exact = exact_solution(x_vals)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_exact, label='Exact Solution', color='black', linewidth=2)
plt.plot(x_e, y_e, 'o-', label='Euler', color='red')
plt.plot(x_me, y_me, 's-', label='Modified Euler (Heun)', color='blue')
plt.plot(x_rk, y_rk, '^-', label='Runge-Kutta 4', color='green')
plt.title("Numerical Methods for ODE Visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
