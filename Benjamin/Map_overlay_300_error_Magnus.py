"""
Map Overlay — Validation Results Heatmap (Absolute Error) — 300m dataset
Reads validation_results_300.txt and plots each validated tile on the map,
colored from light (good prediction) to deep red (large absolute error).

Note: The 300m filenames encode grid indices (not lat/lon),
so the reference map is stretched to fit the grid coordinate bounding box.
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
RESULTS_PATH   = r"output/validation_results_300.txt"
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
errors      = np.abs(predictions - true_counts)

print(f"Absolute error — median: {np.median(errors):.2f}, "
      f"90th pct: {np.percentile(errors, 90):.2f}, "
      f"max: {errors.max():.2f}")

# ── 2. Extract coordinates from filenames ─────────────────────
# 300m filename format: ID_X_Y_count  (e.g. 000001_2167_20401_0)
coords = []
for fname in filenames:
    parts = os.path.splitext(fname)[0].split("_")
    coords.append((float(parts[1]), float(parts[2])))   # (grid_x, grid_y)
coords = np.array(coords)

# ── 3. Set up colormap (transparent at 0, opaque at higher errors) ──
base_cmap = mcolors.LinearSegmentedColormap.from_list(
    "error_heatmap_base",
    ["#f8f7f3", "#d62728"],
    N=256,
)
cmap_colors = base_cmap(np.linspace(0, 1, 256))
fade_end = 25                                            # bottom ~10% fades in from transparent
cmap_colors[:fade_end, 3] = np.linspace(0, 1, fade_end)
cmap = mcolors.ListedColormap(cmap_colors)

max_err = 50.0                                           # cap colorbar at 50 people
norm    = mcolors.Normalize(vmin=0, vmax=max_err)

# Sort so high-error points draw LAST (on top)
order  = np.argsort(errors)
coords = coords[order]
errors = errors[order]

# Mask: only plot tiles with actual errors
nonzero = errors > 0

# ── 4. Load background map ───────────────────────────────────
map_img = plt.imread(MAP_IMAGE_PATH)

x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min,  y_max  = coords[:, 1].min(), coords[:, 1].max()

margin_x = (x_max - x_min) * 0.02
margin_y  = (y_max  - y_min)  * 0.02
extent = [x_min - margin_x, x_max + margin_x,
          y_min  - margin_y,  y_max  + margin_y]

# ── 5. Side-by-side: reference + scatter heatmap ─────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Grid X")
axes[0].set_ylabel("Grid Y")
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
axes[1].set_xlabel("Grid X")
axes[1].set_ylabel("Grid Y")
axes[1].set_title("Absolute Prediction Error Heatmap (Validation Set)", fontsize=13)

cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_absolute_error_heatmap_300.png"), dpi=150)
plt.close()
print("Saved map_absolute_error_heatmap_300.png")

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
ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")
ax.set_title("Absolute Prediction Error — Overlaid on Map", fontsize=14)

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_absolute_error_overlay_300.png"), dpi=150)
plt.close()
print("Saved map_absolute_error_overlay_300.png")

# ── 7. Smooth interpolated heatmap (shared computation) ──────
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

GRID_RES = 500
grid_x = np.linspace(x_min, x_max, GRID_RES)
grid_y = np.linspace(y_min,  y_max,  GRID_RES)
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
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    cmap=cmap,
    norm=norm,
    alpha=0.9,
    aspect="auto",
    interpolation="bilinear",
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Grid X")
ax.set_ylabel("Grid Y")
ax.set_title("Absolute Prediction Error — Smooth Heatmap", fontsize=14)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_absolute_error_smooth_300.png"), dpi=150)
plt.close()
print("Saved map_absolute_error_smooth_300.png")

# ── 7b. Side-by-side: reference + smooth heatmap ────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Grid X")
axes[0].set_ylabel("Grid Y")
axes[0].set_title("Reference Map", fontsize=13)

axes[1].imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)
im = axes[1].imshow(
    grid_err_smooth,
    extent=[x_min, x_max, y_min, y_max],
    origin="lower",
    cmap=cmap,
    norm=norm,
    alpha=0.45,
    aspect="auto",
    interpolation="bilinear",
)
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(extent[2], extent[3])
axes[1].set_xlabel("Grid X")
axes[1].set_ylabel("Grid Y")
axes[1].set_title("Absolute Prediction Error — Smooth Heatmap", fontsize=13)

cbar = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_absolute_error_smooth_sidebyside_300.png"), dpi=150)
plt.close()
print("Saved map_absolute_error_smooth_sidebyside_300.png")