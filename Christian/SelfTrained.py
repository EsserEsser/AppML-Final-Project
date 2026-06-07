import os
import re
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm

print(f"PyTorch {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURE THESE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA_DIR    = "/home/chril/AppML/jpegs_all_300m"   # ← change this
IMG_SIZE    = 600
BATCH_SIZE  = 32                       # Reduced from 96 — 600×600 images × 8 workers was exhausting WSL2 RAM
NUM_SAMPLES = -1
EPOCHS      = 400
LR          = 1e-4
VAL_SPLIT   = 0.2
DROPOUT     = 0.3
PATIENCE    = 100       # early stopping patience
OUTPUT_DIR  = "output"
SEED        = 42
USE_AMP     = True       # Mixed precision training
PATCH_SIZE  = 20        # Larger patches = fewer tokens (600/20 = 30×30 = 900 patches)
EMBED_DIM   = 192        # Increased from 96 to 192
DEPTH       = 6          # Increased from 3 to 6 transformer layers
NUM_HEADS   = 3          # Increased from 2 to 3
NUM_WORKERS = 4          # Reduced from 8 — each worker pins a full batch in RAM
N_WEIGHT_BINS = 10       # Bins for WeightedRandomSampler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()




def extract_target_from_filename(filename: str) -> int:
    """Return the last integer found in the filename (before extension)."""
    stem = Path(filename).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(f"No numeric target found in filename: {filename}")
    return int(numbers[-1])


class PeopleCountDataset(Dataset):
    """Loads images from a directory; target = last number in filename."""

    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        for f in sorted(self.root_dir.iterdir()):
            if f.suffix.lower() in self.VALID_EXT:
                try:
                    target = extract_target_from_filename(f.name)
                    self.samples.append((f, target))
                except ValueError:
                    print(f"  ⚠ Skipping {f.name} — no target in name")

        if not self.samples:
            raise FileNotFoundError(
                f"No valid images in {root_dir}. Check the path and filename format."
            )

        targets = [t for _, t in self.samples]
        print(f"✓ Loaded {len(self.samples)} images")
        print(f"  Target range: {min(targets)} – {max(targets)}  "
              f"(mean {np.mean(targets):.1f}, std {np.std(targets):.1f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(np.log1p(target), dtype=torch.float32)
    
    
    
    
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])





# Load full dataset
full_dataset = PeopleCountDataset(DATA_DIR, transform=None)
full_dataset.samples = full_dataset.samples[:NUM_SAMPLES]

# Exclude zero-population images
full_dataset.samples = [s for s in full_dataset.samples if s[1] > 0]
print(f"After filtering zeros: {len(full_dataset.samples)} samples")

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_subset, val_subset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_dataset = PeopleCountDataset(DATA_DIR, transform=train_transform)
train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]

val_dataset = PeopleCountDataset.__new__(PeopleCountDataset)
val_dataset.root_dir = full_dataset.root_dir
val_dataset.transform = val_transform
val_dataset.samples = [full_dataset.samples[i] for i in val_subset.indices]

train_targets = [t for _, t in train_dataset.samples]
bin_edges = np.histogram_bin_edges(train_targets, bins=N_WEIGHT_BINS)
bin_indices = np.digitize(train_targets, bin_edges[1:-1])
bin_counts = np.bincount(bin_indices, minlength=N_WEIGHT_BINS)
bin_weights = 1.0 / (bin_counts + 1e-6)
sample_weights = torch.tensor([bin_weights[b] for b in bin_indices], dtype=torch.float32)
train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True,
    persistent_workers=True,
)

print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")




targets = [t for _, t in full_dataset.samples]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(targets, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
axes[0].set_xlabel("People Count")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Target Distribution")
axes[0].axvline(np.mean(targets), color="red", linestyle="--", label=f"Mean = {np.mean(targets):.1f}")
axes[0].legend()

# Show a few sample images
sample_indices = random.sample(range(len(full_dataset)), min(6, len(full_dataset)))
axes[1].set_visible(False)

fig2, axes2 = plt.subplots(1, min(6, len(full_dataset)), figsize=(18, 3))
if not isinstance(axes2, np.ndarray):
    axes2 = [axes2]
for ax, idx in zip(axes2, sample_indices):
    path, target = full_dataset.samples[idx]
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    ax.imshow(img)
    ax.set_title(f"Target: {target}")
    ax.axis("off")

plt.tight_layout()
plt.show()




class PatchEmbedding(nn.Module):
    def __init__(self, img_size=200, patch_size=20, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 600/20 → 30×30 = 900 patches

    def forward(self, x):
        x = self.proj(x)               # (B, embed_dim, 30, 30)
        return x.flatten(2).transpose(1, 2)  # (B, 900, embed_dim)


class PeopleCountViT(nn.Module):
    def __init__(self, img_size=200, patch_size=20, embed_dim=192,
                 depth=6, n_heads=3, mlp_ratio=4, dropout=0.1):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2  # 900 for 600×600

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop    = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm        = nn.LayerNorm(embed_dim)
        self.regressor   = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, 1))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x[:, 0])         # CLS token
        return self.regressor(x).squeeze(-1)


model = PeopleCountViT(img_size=IMG_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
                       depth=DEPTH, n_heads=NUM_HEADS, dropout=DROPOUT).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")




criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=1000
)





# MPS workaround: some ops need float32 fallback
if device.type == "mps":
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc="  Training", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device.type, enabled=USE_AMP and device.type == "cuda", dtype=torch.float16):
            preds = model(imgs)
            loss = criterion(preds, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(all_targets, all_preds)
    return epoch_loss, mae


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        with torch.amp.autocast(device.type, enabled=USE_AMP and device.type == "cuda", dtype=torch.float16):
            preds = model(imgs)
            loss = criterion(preds, targets)

        running_loss += loss.item() * imgs.size(0)
        # expm1 back to original scale for MAE/R²
        all_preds.extend(np.expm1(preds.cpu().numpy()))
        all_targets.extend(np.expm1(targets.cpu().numpy()))

    epoch_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return epoch_loss, mae, r2, np.array(all_preds), np.array(all_targets)




best_val_loss = float("inf")
patience_counter = 0
history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")

for epoch in pbar:
    train_loss, train_mae = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
    val_loss, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_mae"].append(train_mae)
    history["val_mae"].append(val_mae)

    pbar.set_postfix({
        "epoch": f"{epoch}/{EPOCHS}",
        "t_loss": f"{train_loss:.4f}",
        "v_loss": f"{val_loss:.4f}",
        "v_mae": f"{val_mae:.2f}",
        "v_r2": f"{val_r2:.3f}",
    })

    # Early stopping + checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_mae": val_mae,
        }, os.path.join(OUTPUT_DIR, "best_model.pth"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹ Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

print(f"\n✓ Training complete — best val loss: {best_val_loss:.4f}")






fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history["train_mae"], label="Train MAE", linewidth=2)
axes[1].plot(history["val_mae"], label="Val MAE", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean Absolute Error")
axes[1].set_title("MAE Curves")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.show()






# Load best checkpoint
ckpt = torch.load(os.path.join(OUTPUT_DIR, "best_model.pth"),
                   map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])

val_loss, val_mae, val_r2, preds, targets = evaluate(model, val_loader, criterion)

print(f"Best model from epoch {ckpt['epoch']}:")
print(f"  Val MSE : {val_loss:.4f}")
print(f"  Val MAE : {val_mae:.2f}")
print(f"  Val R²  : {val_r2:.3f}")





fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(targets, preds, alpha=0.5, edgecolors="k", linewidth=0.3, s=50)
lims = [min(targets.min(), preds.min()) - 1,
        max(targets.max(), preds.max()) + 1]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual People Count", fontsize=12)
ax.set_ylabel("Predicted People Count", fontsize=12)
ax.set_title("Predicted vs Actual", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual.png"), dpi=150)
plt.show()