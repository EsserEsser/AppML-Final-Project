"""
Map Overlay — Validation Results Heatmap (Relative Error)
Reads validation_results.txt and plots each validated tile on the map,
colored from light (good prediction) to deep red (large relative error).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Global font sizes ────────────────────────────────────────
plt.rcParams.update({
    "font.size":        16,
    "axes.titlesize":   20,
    "axes.labelsize":   18,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  14,
})

# ── Config ────────────────────────────────────────────────────
RESULTS_PATH   = r"output/validation_results.txt"
MAP_IMAGE_PATH = r"../Data/Comparison.jpg"
OUTPUT_DIR     = r"output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load validation results ───────────────────────────────
filenames   = []
predictions = []
true_counts = []

with open(RESULTS_PATH, "r") as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split("\t")
        filenames.append(parts[0])
        predictions.append(float(parts[1]))
        true_counts.append(int(parts[3]))

predictions = np.array(predictions)
true_counts = np.array(true_counts)
errors      = np.abs(predictions - true_counts) / np.maximum(true_counts, 1)

print(f"Relative error — median: {np.median(errors):.2f}, "
      f"90th pct: {np.percentile(errors, 90):.2f}, "
      f"max: {errors.max():.2f}")

# ── 2. Extract coordinates from filenames ─────────────────────
coords = []
for fname in filenames:
    parts = os.path.splitext(fname)[0].split("_")
    coords.append((float(parts[3]), float(parts[2])))
coords = np.array(coords)

# ── 3. Set up colormap ───────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "error_heatmap",
    ["#f3f9f6", "#f5e642", "#d62728"],
    N=256,
)
max_err = 2.0                                            # 200% relative error saturates to red
norm    = mcolors.Normalize(vmin=0, vmax=max_err)

# Sort so high-error points draw LAST (on top)
order  = np.argsort(errors)
coords = coords[order]
errors = errors[order]

# Mask: only plot tiles with actual errors
nonzero = errors > 0

# ── 4. Load background map ───────────────────────────────────
map_img = plt.imread(MAP_IMAGE_PATH)

long_min, long_max = coords[:, 0].min(), coords[:, 0].max()
lat_min,  lat_max  = coords[:, 1].min(), coords[:, 1].max()

margin_long = (long_max - long_min) * 0.02
margin_lat  = (lat_max  - lat_min)  * 0.02
extent = [long_min - margin_long, long_max + margin_long,
          lat_min  - margin_lat,  lat_max  + margin_lat]

# ── 5. Side-by-side: reference + scatter heatmap ─────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Reference Map", fontsize=13)

sc = axes[1].scatter(
    coords[nonzero, 0], coords[nonzero, 1],
    c=errors[nonzero],
    cmap=cmap,
    norm=norm,
    s=12, marker="s", alpha=0.7,
    linewidths=0,
    rasterized=True,
)
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(extent[2], extent[3])
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Relative Prediction Error Heatmap (Validation Set)", fontsize=13)

cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label("Relative Error (|pred − true| / true)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_relative_error_heatmap.png"), dpi=150)
plt.close()
print("Saved map_relative_error_heatmap.png")

# ── 6. Overlay: map + scatter heatmap combined ───────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=0.9)

sc = ax.scatter(
    coords[nonzero, 0], coords[nonzero, 1],
    c=errors[nonzero],
    cmap=cmap,
    norm=norm,
    s=12, marker="s", alpha=0.6,
    linewidths=0,
    rasterized=True,
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Relative Prediction Error — Overlaid on Map", fontsize=14)

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Relative Error (|pred − true| / true)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_relative_error_overlay.png"), dpi=150)
plt.close()
print("Saved map_relative_error_overlay.png")

# ── 7. Smooth interpolated heatmap (shared computation) ──────
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

GRID_RES = 500
grid_x = np.linspace(long_min, long_max, GRID_RES)
grid_y = np.linspace(lat_min,  lat_max,  GRID_RES)
gx, gy = np.meshgrid(grid_x, grid_y)

grid_err = griddata(coords, errors, (gx, gy), method="linear")

SIGMA = 3
grid_err_smooth = gaussian_filter(
    np.nan_to_num(grid_err, nan=0.0),
    sigma=SIGMA,
)

nan_mask = gaussian_filter(
    (~np.isnan(grid_err)).astype(float),
    sigma=SIGMA,
)
grid_err_smooth = np.where(nan_mask > 0.1, grid_err_smooth / np.maximum(nan_mask, 1e-6), np.nan)

grid_err_smooth = np.ma.masked_less(grid_err_smooth, 0.05)

# ── 7a. Standalone smooth heatmap ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)

im = ax.imshow(
    grid_err_smooth,
    extent=[long_min, long_max, lat_min, lat_max],
    origin="lower",
    cmap=cmap,
    norm=norm,
    alpha=0.9,
    aspect="auto",
    interpolation="bilinear",
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Relative Prediction Error — Smooth Heatmap", fontsize=14)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Relative Error (|pred − true| / true)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_relative_error_smooth.png"), dpi=150)
plt.close()
print("Saved map_relative_error_smooth.png")

# ── 7b. Side-by-side: reference + smooth heatmap ────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Reference Map", fontsize=13)

axes[1].imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)
im = axes[1].imshow(
    grid_err_smooth,
    extent=[long_min, long_max, lat_min, lat_max],
    origin="lower",
    cmap=cmap,
    norm=norm,
    alpha=0.45,
    aspect="auto",
    interpolation="bilinear",
)
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(extent[2], extent[3])
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Relative Prediction Error — Smooth Heatmap", fontsize=13)

cbar = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Relative Error (|pred − true| / true)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_relative_error_smooth_sidebyside.png"), dpi=150)
plt.close()
print("Saved map_relative_error_smooth_sidebyside.png")