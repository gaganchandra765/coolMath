import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline, PchipInterpolator, UnivariateSpline

def plot_curves_from_points(points):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]

    # Sort points by x for interpolation
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    # Query x range
    xq = np.linspace(x.min(), x.max(), 300)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    # 1. Cubic Spline
    cs = CubicSpline(x, y)
    axes[0].plot(xq, cs(xq), 'b-', label="CubicSpline")
    axes[0].scatter(x, y, c='red')
    axes[0].set_title("Cubic Spline")

    # 2. B-Spline (make_interp_spline)
    bs = make_interp_spline(x, y, k=3)
    axes[1].plot(xq, bs(xq), 'g-', label="B-spline")
    axes[1].scatter(x, y, c='red')
    axes[1].set_title("B-spline")

    # 3. PCHIP
    pchip = PchipInterpolator(x, y)
    axes[2].plot(xq, pchip(xq), 'm-', label="PCHIP")
    axes[2].scatter(x, y, c='red')
    axes[2].set_title("PCHIP (shape-preserving)")

    # 4. Smoothing spline
    us = UnivariateSpline(x, y, s=1)
    axes[3].plot(xq, us(xq), 'c-', label="UnivariateSpline")
    axes[3].scatter(x, y, c='red')
    axes[3].set_title("Univariate Spline (smoothed)")

    for ax in axes:
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Click points on the plot. Press ENTER when done.")
    plt.figure()
    plt.title("Click points, press Enter when done")
    pts = plt.ginput(n=-1, timeout=0)  # unlimited clicks until Enter
    plt.close()

    if len(pts) >= 3:
        plot_curves_from_points(pts)
    else:
        print("Need at least 3 points for spline fitting.")
