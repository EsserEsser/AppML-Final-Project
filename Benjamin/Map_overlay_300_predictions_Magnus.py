"""
Map Overlay — Validation Results Heatmap (Predicted Count) — 300m dataset
Reads validation_results_300.txt and plots each validated tile on the map,
colored from transparent (0 people) to deep green (high predicted count).

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

print(f"Predictions — median: {np.median(predictions):.2f}, "
      f"90th pct: {np.percentile(predictions, 90):.2f}, "
      f"max: {predictions.max():.2f}")

# ── 2. Extract coordinates from filenames ─────────────────────
# 300m filename format: ID_X_Y_count  (e.g. 000001_2167_20401_0)
coords = []
for fname in filenames:
    parts = os.path.splitext(fname)[0].split("_")
    coords.append((float(parts[1]), float(parts[2])))   # (grid_x, grid_y)
coords = np.array(coords)

# ── 3. Set up colormap (transparent at 0, opaque green at higher counts) ──
base_cmap = mcolors.LinearSegmentedColormap.from_list(
    "pred_heatmap_base",
    ["#b8d5b6", "#126512"],
    N=256,
)
cmap_colors = base_cmap(np.linspace(0, 1, 256))
fade_end = 25                                            # bottom ~10% fades in from transparent
cmap_colors[:fade_end, 3] = np.linspace(0, 1, fade_end)
cmap = mcolors.ListedColormap(cmap_colors)

max_val = 100.0                                          # cap colorbar at 100 people
norm    = mcolors.Normalize(vmin=0, vmax=max_val)

# Sort so high-count points draw LAST (on top)
order  = np.argsort(predictions)
coords = coords[order]
predictions = predictions[order]

# Mask: only plot tiles with nonzero predictions
nonzero = predictions > 0

# ── 4. Load background map & compute extents ─────────────────
map_img = plt.imread(MAP_IMAGE_PATH)

x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
y_min, y_max  = coords[:, 1].min(), coords[:, 1].max()

margin_x = (x_max - x_min) * 0.02
margin_y = (y_max - y_min) * 0.02
extent = [x_min - margin_x, x_max + margin_x,
          y_min - margin_y, y_max + margin_y]

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
    c=predictions[nonzero],
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
axes[1].set_title("Predicted People Count (Validation Set)", fontsize=13)

cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label("Predicted Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_prediction_heatmap_300.png"), dpi=150)
plt.close()
print("Saved map_prediction_heatmap_300.png")

# ── 6. Overlay: map + scatter heatmap combined ───────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=0.9)

sc = ax.scatter(
    coords[nonzero, 0], coords[nonzero, 1],
    c=predictions[nonzero],
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
ax.set_title("Predicted People Count — Overlaid on Map", fontsize=14)

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Predicted Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_prediction_overlay_300.png"), dpi=150)
plt.close()
print("Saved map_prediction_overlay_300.png")

# ── 7. Smooth interpolated heatmap (shared computation) ──────
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

GRID_RES = 500
# Use full extent (including margins) so heatmap covers the entire background
grid_x = np.linspace(extent[0], extent[1], GRID_RES)
grid_y = np.linspace(extent[2], extent[3], GRID_RES)
gx, gy = np.meshgrid(grid_x, grid_y)

grid_pred = griddata(coords, predictions, (gx, gy), method="linear")
# Fill NaN edges (outside convex hull) with nearest-neighbor extrapolation
nan_edges = np.isnan(grid_pred)
if nan_edges.any():
    grid_pred[nan_edges] = griddata(
        coords, predictions, (gx[nan_edges], gy[nan_edges]), method="nearest"
    )

SIGMA = 3
grid_pred_smooth = gaussian_filter(
    np.nan_to_num(grid_pred, nan=0.0),
    sigma=SIGMA,
)

nan_mask = gaussian_filter(
    (~np.isnan(grid_pred)).astype(float),
    sigma=SIGMA,
)
grid_pred_smooth = np.where(nan_mask > 0.1, grid_pred_smooth / np.maximum(nan_mask, 1e-6), np.nan)

grid_pred_smooth = np.ma.masked_less(grid_pred_smooth, 0.05)

# ── 7a. Standalone smooth heatmap ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)

im = ax.imshow(
    grid_pred_smooth,
    extent=extent,
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
ax.set_title("Predicted  Count (300m)", fontsize=14)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Predicted Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_prediction_smooth_300.png"), dpi=150)
plt.close()
print("Saved map_prediction_smooth_300.png")

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
    grid_pred_smooth,
    extent=extent,
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
axes[1].set_title("Predicted People Count — Smooth Heatmap", fontsize=13)

cbar = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02, extend="max")
cbar.set_label("Predicted Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_prediction_smooth_sidebyside_300.png"), dpi=150)
plt.close()
print("Saved map_prediction_smooth_sidebyside_300.png")