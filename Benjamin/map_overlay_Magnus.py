"""
Map Overlay — Population Estimate Heatmap
Reads validation_results.txt and plots each validated tile on the map,
colored on a green scale by predicted population count.
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
MAP_IMAGE_PATH = r"../Data/Comparison.jpg"          # ← UPDATE THIS
OUTPUT_DIR     = r"output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load validation results ───────────────────────────────
filenames   = []
predictions = []
true_counts = []

with open(RESULTS_PATH, "r") as f:
    header = f.readline()                                # skip header
    for line in f:
        parts = line.strip().split("\t")
        filenames.append(parts[0])
        predictions.append(float(parts[1]))              # raw prediction
        true_counts.append(int(parts[3]))

predictions = np.array(predictions)
true_counts = np.array(true_counts)

# ── 2. Extract coordinates from filenames ─────────────────────
#    Assumes format:  ..._lat_long_... with lat at index 2, long at index 3
coords = []
for fname in filenames:
    parts = os.path.splitext(fname)[0].split("_")
    coords.append((float(parts[3]), float(parts[2])))    # (long, lat)
coords = np.array(coords)

# ── 3. Set up green colormap ─────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    "population_heatmap",
    ["#acc2ac", "#5cb85c", "#0c450c"],                   # light green → mid green → dark green
    N=256,
)
max_pop = max(np.percentile(predictions, 99.5), 1)         # 95th pctl cap
norm    = mcolors.Normalize(vmin=0, vmax=max_pop)

# Sort so high-population points draw LAST (on top)
order       = np.argsort(predictions)
coords      = coords[order]
predictions = predictions[order]

# Mask: only plot tiles with nonzero population
nonzero = predictions > 0

# ── 4. Load background map ───────────────────────────────────
map_img = plt.imread(MAP_IMAGE_PATH)

long_min, long_max = coords[:, 0].min(), coords[:, 0].max()
lat_min,  lat_max  = coords[:, 1].min(), coords[:, 1].max()

margin_long = (long_max - long_min) * 0.02
margin_lat  = (lat_max  - lat_min)  * 0.02
extent = [long_min - margin_long, long_max + margin_long,
          lat_min  - margin_lat,  lat_max  + margin_lat]

# ── 5. Plot: side-by-side (reference + heatmap) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# Left: plain reference map
axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Reference Map", fontsize=13)

# Right: population estimate scatter
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
axes[1].set_title("Population Estimate Heatmap (Validation Set)", fontsize=13)

cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label("Predicted Population Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_population_heatmap.png"), dpi=150)
plt.close()
print("Saved map_population_heatmap.png")

# ── 6. Plot: overlay (map + heatmap combined) ────────────────
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
ax.set_title("Population Estimate — Overlaid on Map", fontsize=14)

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Predicted Population Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_population_overlay.png"), dpi=150)
plt.close()
print("Saved map_population_overlay.png")

# ── 7. Smooth interpolated heatmap (shared computation) ───────
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Build a fine regular grid across the coordinate range
GRID_RES = 500                                           # resolution of the interpolation grid
grid_x = np.linspace(long_min, long_max, GRID_RES)
grid_y = np.linspace(lat_min,  lat_max,  GRID_RES)
gx, gy = np.meshgrid(grid_x, grid_y)

# Interpolate scattered predictions onto the regular grid
grid_pop = griddata(coords, predictions, (gx, gy), method="linear")

# Apply Gaussian blur for smooth gradients between tiles
SIGMA = 3                                                # smoothing strength — increase for softer look
grid_pop_smooth = gaussian_filter(
    np.nan_to_num(grid_pop, nan=0.0),                    # fill gaps with 0 (no population assumed)
    sigma=SIGMA,
)

# Mask out areas with no nearby data (keep NaN regions transparent)
nan_mask = gaussian_filter(
    (~np.isnan(grid_pop)).astype(float),                 # 1 where data exists, 0 where not
    sigma=SIGMA,
)
grid_pop_smooth = np.where(nan_mask > 0.1, grid_pop_smooth / np.maximum(nan_mask, 1e-6), np.nan)

# Make near-zero population regions transparent (show raw map underneath)
grid_pop_smooth = np.ma.masked_less(grid_pop_smooth, 0.5)

# ── 7a. Standalone smooth heatmap ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)

im = ax.imshow(
    grid_pop_smooth,
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
ax.set_title("Population Estimate — Smooth Heatmap", fontsize=14)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, extend='max')
cbar.set_label("Predicted Population Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_population_smooth.png"), dpi=150)
plt.close()
print("Saved map_population_smooth.png")

# ── 7b. Side-by-side: reference + smooth heatmap ─────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 9))

# Left: plain reference map
axes[0].imshow(map_img, extent=extent, aspect="auto", origin="upper")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("Reference Map", fontsize=13)

# Right: smooth heatmap overlaid on map
axes[1].imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=1.0)
im = axes[1].imshow(
    grid_pop_smooth,
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
axes[1].set_title("Population Estimate — Smooth Heatmap", fontsize=13)

cbar = fig.colorbar(im, ax=axes[1], shrink=0.8, pad=0.02, extend='max')
cbar.set_label("Predicted Population Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_population_smooth_sidebyside.png"), dpi=150)
plt.close()
print("Saved map_population_smooth_sidebyside.png")