import numpy as np
import matplotlib.pyplot as plt

def digitize_curve(y_vals):
    mean_val = np.mean(y_vals)
    binary = (y_vals > mean_val).astype(int)
    return mean_val, binary

if __name__ == "__main__":
    # Load curve data
    curves = np.load("curves.npy", allow_pickle=True).item()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, (name, (x, y)) in enumerate(curves.items()):
        mean_val, binary = digitize_curve(y)
        axes[i].plot(x, y, label=f"{name} curve")
        axes[i].hlines(mean_val, x.min(), x.max(), colors="red", linestyles="--", label="mean")
        axes[i].scatter(x, binary, s=10, c="black", label="digitized (0/1)")
        axes[i].set_title(name)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
