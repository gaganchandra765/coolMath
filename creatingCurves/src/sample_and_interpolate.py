import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import pandas as pd

# Fallback for Lagrange interpolation if scipy is unavailable
use_scipy = True
try:
    from scipy.interpolate import lagrange
except Exception as e:
    use_scipy = False
    from numpy import poly1d as poly1d_from_polyfit
    print("scipy.interpolate.lagrange not available; falling back to numpy.polyfit-based interpolation.")

# Load curves from curves.npy
curves = np.load("curves.npy", allow_pickle=True).item()

# Sampling frequencies to test (Hz)
fs_list = [5,10,15,20]  # Four sampling rates
results = []

# Process each curve type and create a single figure with 2x2 subplots
for curve_name in curves.keys():
    xq, yq = curves[curve_name]  # Original "dense" curve
    t_min, t_max = xq.min(), xq.max()
    t_dense = xq  # Use the original xq as the dense grid
    y_dense = yq

    # Create a 2x2 subplot figure, matching curve_generator.py's style
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, fs in enumerate(fs_list):
        # Sample points
        n_samples = int(fs * (t_max - t_min))  # Samples over the time interval
        t_samples = np.linspace(t_min, t_max, n_samples, endpoint=False)
        y_samples = np.interp(t_samples, xq, yq)

        # Build Lagrange interpolation polynomial
        if use_scipy:
            poly = lagrange(t_samples, y_samples)
            y_recon = poly(t_dense)
        else:
            coeffs = np.polyfit(t_samples, y_samples, deg=n_samples-1)
            poly = poly1d_from_polyfit(coeffs)
            y_recon = poly(t_dense)

        # Compute MSE
        mse = np.mean((y_recon - y_dense)**2)

        # Store results
        results.append({
            "curve": curve_name,
            "fs": fs,
            "n_samples": n_samples,
            "t_samples": t_samples,
            "y_samples": y_samples,
            "y_recon": y_recon,
            "mse": mse
        })

        # Plot on the corresponding subplot
        axes[i].plot(t_dense, y_dense, label=f"Original {curve_name}")
        axes[i].plot(t_dense, y_recon, linestyle='--', label=f"Lagrange recon (fs={fs} Hz)")
        axes[i].scatter(t_samples, y_samples, marker='o', s=40, c="red", label="Samples")
        axes[i].set_title(f"{curve_name}, fs={fs} Hz, samples={n_samples}")
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].legend()
        axes[i].grid(True)

    plt.suptitle(f"Lagrange Interpolation for {curve_name}")
    plt.tight_layout()
    plt.show()

# Plot MSE bar chart for each curve
plt.figure(figsize=(10, 5))
bar_width = 0.2
fs_vals = fs_list
for i, curve_name in enumerate(curves.keys()):
    mses = [r["mse"] for r in results if r["curve"] == curve_name]
    plt.bar([x + i*bar_width for x in range(len(fs_vals))], mses, bar_width, label=curve_name)

plt.xticks([x + bar_width*1.5 for x in range(len(fs_vals))], [str(f) for f in fs_vals])
plt.title("MSE of Lagrange Reconstruction vs Original Curves")
plt.xlabel("Sampling Rate (Hz)")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()

# Print numerical results
print("Numerical Results (MSE):")
for r in results:
    print(f"Curve: {r['curve']}, fs = {r['fs']} Hz, samples = {r['n_samples']}, MSE = {r['mse']:.6e}")

# Display summary table if pandas is available
try:
    summary = pd.DataFrame({
        "Curve": [r["curve"] for r in results],
        "fs (Hz)": [r["fs"] for r in results],
        "n_samples": [r["n_samples"] for r in results],
        "MSE": [r["mse"] for r in results]
    })
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Lagrange Interpolation Summary", summary)
except Exception:
    print("Pandas or caas_jupyter_tools not available; skipping table display.")
