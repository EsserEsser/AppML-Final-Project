"""
Map Overlay — Validation Results Heatmap
Reads validation_results.txt and plots each validated tile on the map,
colored from deep green (perfect prediction) to deep red (large error).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Config ────────────────────────────────────────────────────
RESULTS_PATH   = r"Benjamin\output\validation_results.txt"
MAP_IMAGE_PATH = r"path\to\your\map_image.jpg"          # ← UPDATE THIS
OUTPUT_DIR     = r"Benjamin\output"
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
errors      = np.abs(predictions - true_counts)

# ── 2. Extract coordinates from filenames ─────────────────────
#    Assumes format:  ..._lat_long_... with lat at index 2, long at index 3
coords = []
for fname in filenames:
    parts = os.path.splitext(fname)[0].split("_")
    coords.append((float(parts[3]), float(parts[2])))    # (long, lat)
coords = np.array(coords)

# ── 3. Set up green-to-red colormap ──────────────────────────
#    0 error  → deep green
#    max error → deep red
#    mid range → yellow
cmap = mcolors.LinearSegmentedColormap.from_list(
    "error_heatmap",
    ["#06772e", "#a8d84f", "#f5e642", "#e8871e", "#d62728"],
    N=256,
)
max_err = max(errors.max(), 1)                           # avoid /0
norm    = mcolors.Normalize(vmin=0, vmax=max_err)

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

# Right: error heatmap scatter
sc = axes[1].scatter(
    coords[:, 0], coords[:, 1],
    c=errors,
    cmap=cmap,
    norm=norm,
    s=4, alpha=0.6,
    rasterized=True,
)
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(extent[2], extent[3])
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].set_title("Prediction Error Heatmap (Validation Set)", fontsize=13)

cbar = fig.colorbar(sc, ax=axes[1], shrink=0.8, pad=0.02)
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_validation_heatmap.png"), dpi=150)
plt.close()
print("Saved map_validation_heatmap.png")

# ── 6. Plot: overlay (map + heatmap combined) ────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(map_img, extent=extent, aspect="auto", origin="upper", alpha=0.5)

sc = ax.scatter(
    coords[:, 0], coords[:, 1],
    c=errors,
    cmap=cmap,
    norm=norm,
    s=4, alpha=0.5,
    rasterized=True,
)
ax.set_xlim(extent[0], extent[1])
ax.set_ylim(extent[2], extent[3])
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Prediction Error — Overlaid on Map", fontsize=14)

cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label("Absolute Error (|pred − true|)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "map_validation_overlay.png"), dpi=150)
plt.close()
print("Saved map_validation_overlay.png")
