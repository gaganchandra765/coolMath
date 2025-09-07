import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, make_interp_spline, PchipInterpolator, UnivariateSpline

def plot_and_save_curves(points, savefile="curves.npy"):
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    xq = np.linspace(x.min(), x.max(), 300)

    # Different curves
    curves = {}
    curves["cubic"] = (xq, CubicSpline(x, y)(xq))
    curves["bspline"] = (xq, make_interp_spline(x, y, k=3)(xq))
    curves["pchip"] = (xq, PchipInterpolator(x, y)(xq))
    curves["spline_smooth"] = (xq, UnivariateSpline(x, y, s=1)(xq))

    # Save curve data for digitizer
    np.save(savefile, curves, allow_pickle=True)

    # Plot all 4
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    titles = list(curves.keys())

    for i, (name, (xx, yy)) in enumerate(curves.items()):
        axes[i].plot(xx, yy, label=name)
        axes[i].scatter(x, y, c="red")
        axes[i].set_title(name)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Click points on the plot. Press ENTER when done.")
    plt.figure()
    plt.title("Click points, press Enter when done")
    pts = plt.ginput(n=-1, timeout=0)
    plt.close()

    if len(pts) >= 3:
        plot_and_save_curves(pts, "curves.npy")
        print("Curves saved to curves.npy")
    else:
        print("Need at least 3 points for spline fitting.")
