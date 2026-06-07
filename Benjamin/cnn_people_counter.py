"""
CNN People Counter — Regression from Images
Train a CNN to estimate how many people live in a picture (200×200 px).
The target is extracted from the last number in each filename
(e.g. house_district5_3.jpg → target = 3).

Run as a script (not a notebook) so NUM_WORKERS > 0 works on Windows.
"""

import os
import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for script mode
import matplotlib.pyplot as plt

import torch
import contextlib
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm
from torch.nn import MSELoss


# ──────────────────────────────────────────────────────────────
#  DEVICE SELECTION
# ──────────────────────────────────────────────────────────────
def _select_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print("CUDA available, GPU count:", torch.cuda.device_count())
        torch.backends.cudnn.benchmark = True
        return dev, True
    try:
        import torch_directml
        dev = torch_directml.device()
        print("Using DirectML device:", dev)
        return dev, False
    except Exception:
        pass
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using device: mps")
        return torch.device("mps"), False
    print("Using device: cpu")
    return torch.device("cpu"), False


# ──────────────────────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────────────────────
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

        tgts = [t for _, t in self.samples]
        print(f"✓ Loaded {len(self.samples)} images")
        print(f"  Target range: {min(tgts)} – {max(tgts)}  "
              f"(mean {np.mean(tgts):.1f}, std {np.std(tgts):.1f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(target, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
#  MODELS
# ──────────────────────────────────────────────────────────────
class PeopleCountCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.regressor(self.pool(self.features(x))).squeeze(-1)


class TunableCNN_reg(nn.Module):
    """CNN whose architecture is fully controlled by keyword arguments."""

    def __init__(
        self,
        num_conv_blocks: int = 4,
        base_filters: int = 32,
        filter_scale: float = 2.0,
        double_conv: bool = True,
        fc_hidden: int = 128,
        use_extra_fc: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()

        def _block(in_c, out_c, double):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if double:
                layers += [
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        conv_blocks = []
        in_channels = 1
        out_channels = base_filters
        for _ in range(num_conv_blocks):
            out_channels = min(int(out_channels), 512)
            conv_blocks.append(_block(in_channels, out_channels, double_conv))
            in_channels = out_channels
            out_channels = int(out_channels * filter_scale)

        self.features = nn.Sequential(*conv_blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

        head_layers = [
            nn.Flatten(),
            nn.Linear(in_channels, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        ]
        if use_extra_fc:
            head_layers += [
                nn.Linear(fc_hidden, fc_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            head_layers.append(nn.Linear(fc_hidden // 2, 1))
        else:
            head_layers.append(nn.Linear(fc_hidden, 1))

        self.regressor = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.regressor(self.pool(self.features(x))).squeeze(-1)





# ──────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, use_amp, scaler, autocast):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc="  Training", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(all_targets, all_preds)
    return epoch_loss, mae


@torch.no_grad()
def evaluate(model, loader, criterion, device, autocast):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, targets)

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return epoch_loss, mae, r2, np.array(all_preds), np.array(all_targets)


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    # ── Configuration ─────────────────────────────────────────
    DATA_DIR    = r"C:\Users\Ben12\Documents\GitHub\AppML-Final-Project\Data\Data_All"
    IMG_SIZE    = 200
    NUM_SAMPLES = -1
    OUTPUT_DIR  = "output"
    VAL_SPLIT   = 0.2
    SEED        = 42
    NUM_WORKERS = 2

    #REGRESSOR parameters
    BATCH_SIZE  = 32
    EPOCHS      = 100 #How many epochs to train the final model for
    PATIENCE    = 20 #Early stopping patience (in epochs without improvement on val loss)
    LR          = 9.180109604625473e-05
    WEIGHT_DECAY = 0.0004282198922963024
    DROPOUT     = 0.26612494072338866
    NUM_CONV_BLOCKS = 5
    BASE_FILTERS    = 64
    FILTER_SCALE    = 2.0
    DOUBLE_CONV     = True
    FC_HIDDEN       = 256
    USE_EXTRA_FC    = True
    TRAIN_POSITIVE_ONLY = True #If True, only train on samples with target > 0 (i.e. at least 1 person in the image).
    BALANCE_DATASET = False # If True, balance the dataset so that samples with target=0 and target>0 are equally represented. Only has an effect if TRAIN_POSITIVE_ONLY is False.

    # REGRESSOR Optuna controls
    RUN_JOINT_SEARCH = True  # If True, architecture and HP search run together in arch_objective. RUN_ARCH_TUNING must be True for this to have any effect.
    RUN_OPTUNA  = False
    RUN_ARCH_TUNING = True
    OPTUNA_TRIALS = 30
    OPTUNA_EPOCHS = 10
    OPTUNA_TRAIN_SAMPLES = 3000
    OPTUNA_VAL_SAMPLES   = 1000
    ARCH_OPTUNA_TRIALS   = 30
    ARCH_OPTUNA_EPOCHS   = 8
    ARCH_OPTUNA_TRAIN_SAMPLES = 3000
    ARCH_OPTUNA_VAL_SAMPLES   = 1000

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Seed ──────────────────────────────────────────────────
    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    set_seed()

    # ── Device ────────────────────────────────────────────────
    device, use_amp = _select_device()
    PIN_MEMORY = getattr(device, "type", "") == "cuda"
    print("pin_memory enabled:", PIN_MEMORY)

    # MPS workaround
    if getattr(device, "type", "") == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    import torch_directml
    print(torch_directml.device_count())
    for i in range(torch_directml.device_count()):
        print(i, torch_directml.device_name(i))
    device = torch_directml.device(1)

    scaler   = torch.cuda.amp.GradScaler() if use_amp else None
    autocast = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    # ── Transforms ────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ── Dataset ───────────────────────────────────────────────
    full_dataset = PeopleCountDataset(DATA_DIR, transform=None)

    if TRAIN_POSITIVE_ONLY:
        filtered_samples = [s for s in full_dataset.samples if s[1] > 0]
        print(f"Positive-only dataset: {len(filtered_samples)} samples (targets > 0)")
    else:
        filtered_samples = list(full_dataset.samples)
        print(f"Full dataset: {len(filtered_samples)} samples (including zero targets)")

    if BALANCE_DATASET:
        if TRAIN_POSITIVE_ONLY:
            by_target = defaultdict(list)
            for sample in filtered_samples:
                by_target[sample[1]].append(sample)
            min_count = min(len(v) for v in by_target.values())
            balanced_samples = []
            for tv in sorted(by_target):
                random.shuffle(by_target[tv])
                balanced_samples.extend(by_target[tv][:min_count])
            filtered_samples = balanced_samples
            print(f"Balanced positive-only dataset: {len(filtered_samples)} total "
                  f"from {len(by_target)} target values")
        else:
            zero_samples    = [s for s in filtered_samples if s[1] == 0]
            nonzero_samples = [s for s in filtered_samples if s[1] > 0]
            n_balanced = min(len(zero_samples), len(nonzero_samples))
            random.shuffle(zero_samples)
            random.shuffle(nonzero_samples)
            filtered_samples = zero_samples[:n_balanced] + nonzero_samples[:n_balanced]
            random.shuffle(filtered_samples)
            print(f"Balanced: {n_balanced} zero-people + {n_balanced} with-people "
                  f"= {len(filtered_samples)} total")

    full_dataset.samples = filtered_samples
    random.shuffle(full_dataset.samples)
    if NUM_SAMPLES != -1:
        full_dataset.samples = full_dataset.samples[:NUM_SAMPLES]

    val_size   = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_dataset = PeopleCountDataset(DATA_DIR, transform=train_transform)
    train_dataset.samples = [full_dataset.samples[i] for i in train_subset.indices]

    val_dataset = PeopleCountDataset.__new__(PeopleCountDataset)
    val_dataset.root_dir  = full_dataset.root_dir
    val_dataset.transform = val_transform
    val_dataset.samples   = [full_dataset.samples[i] for i in val_subset.indices]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory= False, persistent_workers= True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory= False, persistent_workers= True,
    )
    print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")

    # ── EDA plots ─────────────────────────────────────────────
    tgts = [t for _, t in full_dataset.samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tgts, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("People Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Target Distribution")
    ax.axvline(np.mean(tgts), color="red", linestyle="--",
               label=f"Mean = {np.mean(tgts):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150)
    plt.close()

    # ── Base model (used by HP Optuna) ────────────────────────
    model = PeopleCountCNN(dropout=DROPOUT)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")


    # ── Architecture Optuna ───────────────────────────────────
    if RUN_ARCH_TUNING:
        def arch_objective(trial):
            num_conv_blocks = trial.suggest_int("num_conv_blocks", 3, 7)
            base_filters    = trial.suggest_categorical("base_filters", [32, 64, 128])
            filter_scale    = trial.suggest_categorical("filter_scale", [1.0, 1.5, 2.0, 2.5])
            double_conv     = trial.suggest_categorical("double_conv", [True, False])
            fc_hidden       = trial.suggest_categorical("fc_hidden", [12, 32, 64, 128, 256, 512])
            use_extra_fc    = trial.suggest_categorical("use_extra_fc", [True, False])
            arch_dropout    = trial.suggest_float("arch_dropout", 0.05, 0.7)

            lr_val = LR
            wd_val = WEIGHT_DECAY
            batch_val = BATCH_SIZE

            if RUN_JOINT_SEARCH:
                lr_val = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
                wd_val = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
                batch_val = trial.suggest_categorical("batch_size", [4, 16, 32, 64])

            if num_conv_blocks > 5 and base_filters >= 128:
                raise optuna.exceptions.TrialPruned()
            
        
            m = TunableCNN_reg(
                num_conv_blocks=num_conv_blocks,
                base_filters=base_filters,
                filter_scale=filter_scale,
                double_conv=double_conv,
                fc_hidden=fc_hidden,
                use_extra_fc=use_extra_fc,
                dropout=arch_dropout,
            ).to(device)

            crit = MSELoss()
            opt  = optim.AdamW(m.parameters(), lr=lr_val,
                               weight_decay=wd_val, foreach=False)


            def _loader(dataset, n, shuffle):
                n = min(n, len(dataset))
                if n < len(dataset):
                    subset, _ = random_split(
                        dataset,
                        [n, len(dataset) - n],
                        generator=torch.Generator().manual_seed(SEED),
                    )
                else:
                    subset = dataset
                return DataLoader(
                    subset, batch_size=batch_val, shuffle=shuffle,
                    num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True,
                )

            t_loader = _loader(train_dataset, ARCH_OPTUNA_TRAIN_SAMPLES, True)
            v_loader = _loader(val_dataset,   ARCH_OPTUNA_VAL_SAMPLES,   False)

            best = float("inf")
            for epoch in range(ARCH_OPTUNA_EPOCHS):
                train_one_epoch(m, t_loader, crit, opt,
                                device, use_amp, scaler, autocast)
                val_loss, *_ = evaluate(m, v_loader, crit, device, autocast)
                trial.report(val_loss, epoch)
                if epoch >= max(1, ARCH_OPTUNA_EPOCHS // 2) and trial.should_prune():
                    raise optuna.TrialPruned()
                best = min(best, val_loss)
            return best

        print("Starting architecture search...")
        arch_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=max(1, ARCH_OPTUNA_EPOCHS - 2),
            ),
        )
        arch_study.optimize(
            arch_objective, n_trials=ARCH_OPTUNA_TRIALS, show_progress_bar=True
        )

        best_arch = arch_study.best_params

        if RUN_JOINT_SEARCH:
            LR = best_arch["lr"]
            BATCH_SIZE = best_arch["batch_size"]
            WEIGHT_DECAY = best_arch["weight_decay"]

        print("\nBest architecture params:")
        for k, v in best_arch.items():
            print(f"  {k}: {v}")

        NUM_CONV_BLOCKS = best_arch["num_conv_blocks"]
        BASE_FILTERS = best_arch["base_filters"]
        FILTER_SCALE = best_arch["filter_scale"]
        DOUBLE_CONV = best_arch["double_conv"]
        FC_HIDDEN = best_arch["fc_hidden"]
        USE_EXTRA_FC = best_arch["use_extra_fc"]
        DROPOUT = best_arch["arch_dropout"]

        model = TunableCNN_reg(
            num_conv_blocks=best_arch["num_conv_blocks"],
            base_filters=best_arch["base_filters"],
            filter_scale=best_arch["filter_scale"],
            double_conv=best_arch["double_conv"],
            fc_hidden=best_arch["fc_hidden"],
            use_extra_fc=best_arch["use_extra_fc"],
            dropout=best_arch["arch_dropout"],
        )
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✓ Architecture search complete — model rebuilt ({total_params:,} parameters)")
    else:
        print("Architecture search skipped (RUN_ARCH_TUNING = False).")

    # ── HP Optuna ─────────────────────────────────────────────
    if RUN_OPTUNA:
        def make_subset_loader(dataset, sample_size, batch_size, seed, shuffle):
            sample_size = min(sample_size, len(dataset))
            if sample_size == len(dataset):
                subset = dataset
            else:
                subset, _ = random_split(
                    dataset,
                    [sample_size, len(dataset) - sample_size],
                    generator=torch.Generator().manual_seed(seed),
                )
            return DataLoader(
                subset, batch_size=batch_size, shuffle=shuffle,
                num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True,
            )

        def objective(trial):
            trial_batch_size  = trial.suggest_categorical("batch_size", [4, 16, 32, 64])
            trial_lr          = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
            trial_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

            t_loader = make_subset_loader(train_dataset, OPTUNA_TRAIN_SAMPLES,
                                          trial_batch_size, SEED, True)
            v_loader = make_subset_loader(val_dataset,   OPTUNA_VAL_SAMPLES,
                                          trial_batch_size, SEED, False)

            m = TunableCNN_reg(
            num_conv_blocks= NUM_CONV_BLOCKS,
            base_filters= BASE_FILTERS,
            filter_scale= FILTER_SCALE,
            double_conv=   DOUBLE_CONV,
            fc_hidden= FC_HIDDEN,
            use_extra_fc= USE_EXTRA_FC,
            dropout= DROPOUT,).to(device)

            crit = MSELoss()
            opt  = optim.AdamW(m.parameters(), lr=trial_lr,
                               weight_decay=trial_weight_decay, foreach=False)

            best = float("inf")
            for epoch in range(OPTUNA_EPOCHS):
                train_one_epoch(m, t_loader, crit, opt,
                                device, use_amp, scaler, autocast)
                val_loss, *_ = evaluate(m, v_loader, crit, device, autocast)
                trial.report(val_loss, epoch)
                if epoch >= max(1, OPTUNA_EPOCHS // 2) and trial.should_prune():
                    raise optuna.TrialPruned()
                best = min(best, val_loss)
            return best

        print("Starting HP Optuna search...")
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=15, n_warmup_steps=max(1, OPTUNA_EPOCHS - 1)
            ),
        )
        # Seed trial 0 with the hand-configured starting values
        study.enqueue_trial({
            "batch_size":   BATCH_SIZE,
            "lr":           LR,
            "weight_decay": WEIGHT_DECAY,
        })
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

        best_params  = study.best_params
        print("\nBest HP Optuna params:", best_params)

        BATCH_SIZE   = best_params["batch_size"]
        LR           = best_params["lr"]
        WEIGHT_DECAY = best_params["weight_decay"]

        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=False, persistent_workers=True,
        )


        model = TunableCNN_reg(
            num_conv_blocks= NUM_CONV_BLOCKS,
            base_filters= BASE_FILTERS,
            filter_scale= FILTER_SCALE,
            double_conv=   DOUBLE_CONV,
            fc_hidden= FC_HIDDEN,
            use_extra_fc= USE_EXTRA_FC,
            dropout= DROPOUT,).to(device)
    else:
        print("HP Optuna skipped. Using configured hyperparameters.")
        model = TunableCNN_reg(
            num_conv_blocks= NUM_CONV_BLOCKS,
            base_filters= BASE_FILTERS,
            filter_scale= FILTER_SCALE,
            double_conv=   DOUBLE_CONV,
            fc_hidden= FC_HIDDEN,
            use_extra_fc= USE_EXTRA_FC,
            dropout= DROPOUT,).to(device)

    
    # ── Full training ─────────────────────────────────────────
    criterion = MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss   = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}

    pbar = tqdm(range(1, EPOCHS + 1), desc="Training")
    for epoch in pbar:
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, use_amp, scaler, autocast
        )
        val_loss, val_mae, val_r2, _, _ = evaluate(
            model, val_loader, criterion, device, autocast
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        pbar.set_postfix({
            "t_loss": f"{train_loss:.4f}",
            "v_loss": f"{val_loss:.4f}",
            "v_mae":  f"{val_mae:.2f}",
            "v_r2":   f"{val_r2:.3f}",
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            
            patience_counter = 0
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            val_loss,
                "val_mae":             val_mae,
            }, os.path.join(OUTPUT_DIR, "best_model_regressor.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹ Early stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print(f"\n✓ Training complete — best val loss: {best_val_loss:.4f}")


    # ── Training curves ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"],   label="Val Loss",   linewidth=2)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_mae"], label="Train MAE", linewidth=2)
    axes[1].plot(history["val_mae"],   label="Val MAE",   linewidth=2)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Mean Absolute Error")
    axes[1].set_title("MAE Curves"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    # ── Final evaluation ──────────────────────────────────────
    ckpt_path = os.path.join(OUTPUT_DIR, "best_model_regressor.pth")
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    val_loss, val_mae, val_r2, preds, targets = evaluate(
        model, val_loader, criterion, device, autocast
    )
    print(f"Best model from epoch {ckpt['epoch']}:")
    print(f"  Val MSE : {val_loss:.4f}")
    print(f"  Val MAE : {val_mae:.2f}")
    print(f"  Val R²  : {val_r2:.3f}")

    # Pred vs actual
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.5, edgecolors="k", linewidth=0.3, s=50)
    lims = [min(targets.min(), preds.min()) - 1, max(targets.max(), preds.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual People Count", fontsize=12)
    ax.set_ylabel("Predicted People Count", fontsize=12)
    ax.set_title("Predicted vs Actual", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual.png"), dpi=150)
    plt.close()

    # Error analysis
    errors     = preds - targets
    abs_errors = np.abs(errors)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.7, color="#DD8452")
    axes[0].set_xlabel("Prediction Error (pred − actual)")
    axes[0].set_ylabel("Frequency"); axes[0].set_title("Error Distribution")
    axes[0].axvline(0, color="red", linestyle="--")
    worst_idx = np.argsort(abs_errors)[-10:][::-1]
    axes[1].barh(range(len(worst_idx)), abs_errors[worst_idx],
                 color="#C44E52", edgecolor="black")
    axes[1].set_xlabel("Absolute Error"); axes[1].set_ylabel("Sample Index")
    axes[1].set_title("Top 10 Worst Predictions")
    for i, idx in enumerate(worst_idx):
        axes[1].text(abs_errors[idx] + 0.1, i,
                     f"actual={targets[idx]:.0f}, pred={preds[idx]:.1f}", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_analysis.png"), dpi=150)
    plt.close()

    print(f"Median absolute error: {np.median(abs_errors):.2f}")
    print(f"90th percentile error: {np.percentile(abs_errors, 90):.2f}")
    print(f"Within ±1 person:  {(abs_errors <= 1).mean()*100:.1f}%")
    print(f"Within ±2 people:  {(abs_errors <= 2).mean()*100:.1f}%")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    # Force 'spawn' method which is safer for Windows/DirectML
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
