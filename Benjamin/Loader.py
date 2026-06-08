"""
Classifier + Regressor Pipeline (Dual-Tuned + Full Analytics)
1. Splits data into Train and Validation sets.
2. Generates EDA Target Distribution plot.
3. Tunes & Trains a Classifier to predict if people are present (>0).
4. Tunes & Trains a Regressor to count people (on data where targets > 0) & plots curves.
5. Evaluates pipeline on full Validation set & plots Pred vs Actual + Error Analysis.
"""

import os
import re
import random
from pathlib import Path
from scipy.stats import gaussian_kde

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import contextlib
import optuna
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score, mean_squared_error, precision_score, r2_score, recall_score
from tqdm.auto import tqdm
from torch.nn import MSELoss, BCEWithLogitsLoss

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
    stem = Path(filename).stem
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        raise ValueError(f"No numeric target found in filename: {filename}")
    return int(numbers[-1])

class PeopleCountDataset(Dataset):
    VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root_dir: str = None, transform=None, samples=None, mode="regress"):
        self.transform = transform
        self.mode = mode

        if samples is not None:
            self.samples = list(samples)
        else:
            self.samples = []
            self.root_dir = Path(root_dir)
            for f in sorted(self.root_dir.iterdir()):
                if f.suffix.lower() in self.VALID_EXT:
                    try:
                        target = extract_target_from_filename(f.name)
                        self.samples.append((f, target))
                    except ValueError:
                        print(f"Warning: Skipping file with invalid name format: {f.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform:
            img = self.transform(img)
            
        if self.mode == "classify":
            target = 1.0 if target > 0 else 0.0
        else:
            target = float(target)
            
        return img, torch.tensor(target, dtype=torch.float32)

# ──────────────────────────────────────────────────────────────
#  MODELS
# ──────────────────────────────────────────────────────────────
class TunableCNN(nn.Module):
    def __init__(self, num_conv_blocks=4, base_filters=32, filter_scale=2.0, 
                 double_conv=True, fc_hidden=128, use_extra_fc=False, dropout=0.3):
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
                nn.Linear(fc_hidden // 2, 1)
            ]
        else:
            head_layers.append(nn.Linear(fc_hidden, 1))

        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        return self.head(self.pool(self.features(x))).squeeze(-1)
    

    
class CountCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(1,   32),
            conv_block(32,  64),
            conv_block(64,  128),
            conv_block(128, 256),
            conv_block(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(    # renamed from regressor
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),              # raw logit, no ReLU
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x))).squeeze(1)





# ──────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ──────────────────────────────────────────────────────────────
def train_one_epoch_class(model, loader, criterion, optimizer, device, use_amp, scaler, autocast):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc="  Train Class", leave=False):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)

@torch.no_grad()
def eval_class(model, loader, criterion, device, autocast):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, targets)

        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    return running_loss / len(loader.dataset), accuracy_score(all_targets, all_preds)

def train_one_epoch_reg(model, loader, criterion, optimizer, device, use_amp, scaler, autocast):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc="  Train Reg", leave=False):
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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

    return running_loss / len(loader.dataset), mean_absolute_error(all_targets, all_preds)

@torch.no_grad()
def eval_reg(model, loader, criterion, device, autocast):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for imgs, targets in loader:
        imgs, targets = imgs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with autocast():
            preds = model(imgs)
            loss = criterion(preds, targets)

        running_loss += loss.item() * imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    return running_loss / len(loader.dataset), mean_absolute_error(all_targets, all_preds)

@torch.no_grad()
def evaluate_pipeline(classifier, regressor, loader, device, autocast):
    classifier.eval()
    regressor.eval()
    all_preds, all_targets = [], []

    for imgs, targets in tqdm(loader, desc="  Pipeline Eval"):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.cpu().numpy()
        
        with autocast():
            class_logits = classifier(imgs)
            class_preds = (torch.sigmoid(class_logits) > 0.5).float()
            reg_preds = regressor(imgs)
            
            final_preds = class_preds * reg_preds
            
        all_preds.extend(final_preds.cpu().numpy())
        all_targets.extend(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mse = np.mean((all_preds - all_targets)**2)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.0
    return mse, mae, r2, all_preds, all_targets

# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────
def main():
    DATA_DIR    = r"C:\Users\Ben12\Documents\GitHub\AppML-Final-Project\Data\Data_All"
    IMG_SIZE    = 200
    OUTPUT_DIR  = "Benjamin\\output"
    VAL_SPLIT   = 0.2
    SEED        = 42
    NUM_WORKERS = 2
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Saving plots to:", os.path.abspath(OUTPUT_DIR))


    # Classifier Search Settings
    CLASSIFIER_PTH = r"Benjamin\best_model_full.pth"  # Set to .pth path to skip tuning and training

    REGRESSOR_PTH = r"Benjamin\output\best_reg_Loader.pth"  # Set to .pth path to skip tuning and training

    # Defaults (Overwritten by Optuna if RUN_ARCH_TUNING is True)
    c_arch = {"num_conv_blocks": 4, "base_filters": 32, "filter_scale": 2.0, "double_conv": True, "fc_hidden": 128, "use_extra_fc": False}
    c_hp   = {"lr": 1e-4, "weight_decay": 1e-4, "batch_size": 32, "dropout": 0.3}

    r_arch = {"num_conv_blocks": 5, "base_filters": 64, "filter_scale": 2.0, "double_conv": True, "fc_hidden": 256, "use_extra_fc": True}
    r_hp   = {"lr": 9.180109604625473e-05, "weight_decay": 0.0004282198922963024, "batch_size": 32, "dropout": 0.26612494072338866}



    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    set_seed()

    device, use_amp = _select_device()
    import torch_directml
    print(torch_directml.device_count())
    for i in range(torch_directml.device_count()):
        print(i, torch_directml.device_name(i))
    device = torch_directml.device(1)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    autocast = torch.cuda.amp.autocast if use_amp else contextlib.nullcontext

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    val_transform = transforms.Compose([
        #transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # ── 1. Data Prep & EDA ─────────────────────────────
    print("\n--- Phase 1: Data Preparation ---")
    full_dataset_raw = PeopleCountDataset(DATA_DIR, transform=None)
    #print first 10 entries of full dataset:
    print("Sample entries from Full Dataset:")
    for i in range(10):
        print(f"  {full_dataset_raw.samples[i][0].name}: Target = {full_dataset_raw.samples[i][1]}")
    all_samples = full_dataset_raw.samples

    # EDA: Target Distribution Plot
    tgts = [t for _, t in all_samples]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tgts, bins=30, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("People Count")
    ax.set_ylabel("Frequency")
    ax.set_title("Target Distribution")
    ax.axvline(np.mean(tgts), color="red", linestyle="--", label=f"Mean = {np.mean(tgts):.1f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"), dpi=150)
    plt.close()
    print(f"✓ Saved Target Distribution Plot ({len(all_samples)} total images)")

    random.shuffle(all_samples)
    val_size = int(len(all_samples) * VAL_SPLIT)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]

    train_reg_samples = [s for s in train_samples if s[1] > 0]
    val_reg_samples = [s for s in val_samples if s[1] > 0]

    n_neg = len([s for s in train_samples if s[1] == 0])
    n_pos = len([s for s in train_samples if s[1] > 0])
    pos_weight = torch.tensor([n_neg / n_pos]).to(device) #Punishes false negatives more than false positives instead of balancing dataset

    ds_class_train = PeopleCountDataset(samples=train_samples, transform=train_transform, mode="classify")
    ds_class_val   = PeopleCountDataset(samples=val_samples, transform=val_transform, mode="classify")
    ds_reg_train = PeopleCountDataset(samples=train_reg_samples, transform=train_transform, mode="regress")
    ds_reg_val   = PeopleCountDataset(samples=val_reg_samples, transform=val_transform, mode="regress")

    #print first 10 entries of val dataset:
    print("\nSample entries from Validation Set:")
    for i in range(10):
        print(f"  {val_samples[i][0].name}: Target = {val_samples[i][1]}")


    print("\n--- Phase 2: Classifier ---")
    if CLASSIFIER_PTH is not None:
        # Load pretrained classifier from .pth file — skip Optuna and training entirely
        print(f"Loading pretrained classifier from: {CLASSIFIER_PTH}")

        best = {
            "lr": 8.590847395090266e-05,
            "dropout": 0.0029985914779085687,
            "weight_decay": 0.00017033389914119428,
            "batch_size": 64,
            "optimizer": "Adam"
        }

         #assign best params back to c_hp so they're used downstream
        c_hp["lr"]           = best["lr"]
        c_hp["dropout"]      = best["dropout"]
        c_hp["weight_decay"] = best["weight_decay"]
        c_hp["batch_size"]   = best["batch_size"]
        
        classifier = CountCNN(dropout=best["dropout"]).to(device)
        checkpoint = torch.load(CLASSIFIER_PTH, map_location="cpu", weights_only=False)  # load checkpoint dict
        classifier.load_state_dict(checkpoint["model_state_dict"])                       # extract just the weights
        classifier.to(device)
        classifier.eval()
        c_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        print("✓ Classifier loaded successfully.")
    else:
        #Dont train anythin but still allow script to run safely
        print("No pretrained classifier path provided. Skipping classifier training and using untrained model for pipeline evaluation.")
        classifier = TunableCNN(**c_arch, dropout=c_hp["dropout"]).to(device)
        c_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        print("✓ Classifier loaded successfully.")


    # ── 3. Regressor Optuna ─────────────────────────────
    print("\n--- Phase 3: Tuning Regressor ---")
    if REGRESSOR_PTH is not None:
        # Load pretrained regressor from .pth file — skip Optuna and training entirely
        print(f"Loading pretrained regressor from: {REGRESSOR_PTH}")
        
        regressor = TunableCNN(**r_arch).to(device)
        
        checkpoint = torch.load(REGRESSOR_PTH, map_location="cpu", weights_only=False)  # load checkpoint dict
        regressor.load_state_dict(checkpoint)                     # extract just the weights
        regressor.to(device)
        regressor.eval()
        r_history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
        print("✓ Regressor loaded successfully.")
    else:
        #Dont train anythin but still allow script to run safely
        print("No pretrained regressor path provided. Skipping regressor training and using untrained model for pipeline evaluation.")
        regressor = TunableCNN(**r_arch, dropout=r_hp["dropout"]).to(device)
        r_history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
        print("✓ Regressor loaded successfully.")
    

# ── 4. Pipeline Eval & Analytics ─────────────────────────────
    print("\n--- Phase 4: Pipeline Evaluation ---")
    ds_pipe_val = PeopleCountDataset(samples=val_samples, transform=val_transform, mode="regress")
    dl_classifier_val = DataLoader(ds_pipe_val, batch_size=best["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    dl_regressor_val  = DataLoader(ds_pipe_val, batch_size=r_hp["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    # Gather data for ALL metrics
    classifier.eval()
    regressor.eval()

    # Pass 1: Classifier
    all_class_preds = []
    all_class_targets = []
    with torch.no_grad():
        for imgs, targets in tqdm(dl_classifier_val, desc="  Classifier Pass"):
            imgs = imgs.to(device)
            with autocast():
                c_logits = classifier(imgs)
            c_preds = (torch.sigmoid(c_logits) > 0.5).float()
            all_class_preds.extend(c_preds.cpu().numpy())
            all_class_targets.extend((targets > 0).float().numpy())

    occupied_mask = np.array(all_class_preds).astype(bool)
    truly_occupied_indices = np.where(np.array([t for _, t in ds_pipe_val.samples]) > 0)[0]

    # Pass 2a: Regressor on classifier-passed images
    reg_preds_occupied = []
    ds_occupied = torch.utils.data.Subset(ds_pipe_val, np.where(occupied_mask)[0])
    dl_occupied = DataLoader(ds_occupied, batch_size=r_hp["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    with torch.no_grad():
        for imgs, targets in tqdm(dl_occupied, desc="  Regressor Pass (Classifier-passed)"):
            imgs = imgs.to(device)
            with autocast():
                reg_preds_occupied.extend(regressor(imgs).cpu().numpy())

    # Pass 2b: Regressor on truly occupied images (bypasses classifier entirely)
    reg_preds_isolated = []
    ds_truly_occupied = torch.utils.data.Subset(ds_pipe_val, truly_occupied_indices)
    dl_truly_occupied = DataLoader(ds_truly_occupied, batch_size=r_hp["batch_size"], shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    with torch.no_grad():
        for imgs, targets in tqdm(dl_truly_occupied, desc="  Regressor Pass (Truly Occupied)"):
            imgs = imgs.to(device)
            with autocast():
                reg_preds_isolated.extend(regressor(imgs).cpu().numpy())

    # Combine into final arrays for pipeline metrics
    all_preds = np.zeros(len(ds_pipe_val))
    all_preds[occupied_mask] = reg_preds_occupied
    all_targets = [t for _, t in ds_pipe_val.samples]

    # Isolated regressor metrics (comparable to training val loss)
    reg_targets_isolated = np.array([t for _, t in ds_pipe_val.samples if t > 0])
    reg_preds_isolated   = np.array(reg_preds_isolated)
    reg_mse_isolated = np.mean((reg_preds_isolated - reg_targets_isolated)**2)
    reg_mae_isolated = mean_absolute_error(reg_targets_isolated, reg_preds_isolated)
    reg_r2_isolated  = r2_score(reg_targets_isolated, reg_preds_isolated)

    # Classifier-passed metrics
    classifier_occupied_mask = occupied_mask
    reg_targets_cls = np.array(all_targets)[classifier_occupied_mask]
    reg_preds_cls   = np.array(all_preds)[classifier_occupied_mask]
    reg_mse_cls = np.mean((reg_preds_cls - reg_targets_cls)**2)
    reg_mae_cls = mean_absolute_error(reg_targets_cls, reg_preds_cls)
    reg_r2_cls  = r2_score(reg_targets_cls, reg_preds_cls)

    print("REGRESSOR (Truly Occupied, classifier bypassed — comparable to training val loss):")
    print(f"  MSE: {reg_mse_isolated:.4f}")
    print(f"  MAE: {reg_mae_isolated:.4f}")
    print(f"  R² : {reg_r2_isolated:.4f}")
    print("-" * 30)
    print("REGRESSOR (Classifier-passed, may include false positives):")
    print(f"  MSE: {reg_mse_cls:.4f}")
    print(f"  MAE: {reg_mae_cls:.4f}")
    print(f"  R² : {reg_r2_cls:.4f}")


    # --- Confusion Matrix (Classifier) ---
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(all_class_targets, all_class_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Empty", "Occupied"])
    disp.plot(cmap="Blues", ax=ax)
    plt.title("Classifier Confusion Matrix")
    plt.savefig(os.path.join(OUTPUT_DIR, "classifier_confusion_matrix.png"), dpi=150)
    plt.close()

    # --- Regressor Pred vs Actual ---
    # Masks
    truly_occupied_mask = np.array(all_targets) > 0          # ground truth non-empty
    classifier_occupied_mask = occupied_mask                   # what classifier passed to regressor

    # --- Regressor Plot (only what classifier passed to regressor) ---
    reg_targets_cls = np.array(all_targets)[classifier_occupied_mask]
    reg_preds_cls   = np.array(all_preds)[classifier_occupied_mask]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(reg_targets_cls, reg_preds_cls, alpha=0.5, color="green", edgecolors="k")
    lims = [min(reg_targets_cls.min(), reg_preds_cls.min()), max(reg_targets_cls.max(), reg_preds_cls.max())]
    ax.plot(lims, lims, "r--", label="Perfect")
    ax.set_xlabel("Actual Count")
    ax.set_ylabel("Predicted Count")
    ax.set_title("Regressor Only: Predicted vs Actual (Classifier-passed images)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "regressor_pred_vs_actual.png"), dpi=150)
    plt.close()

    # --- Classifier Metrics ---
    cls_acc = accuracy_score(all_class_targets, all_class_preds)
    cls_pre = precision_score(all_class_targets, all_class_preds, average='binary')
    cls_rec = recall_score(all_class_targets, all_class_preds, average='binary')
    cls_f1  = f1_score(all_class_targets, all_class_preds, average='binary')

    #print Matthews correlation coefficient (MCC) which is a better metric for imbalanced binary classification
    from sklearn.metrics import matthews_corrcoef
    cls_mcc = matthews_corrcoef(all_class_targets, all_class_preds)

    print("CLASSIFIER (Binary Occupancy):")
    print(f"  Accuracy:  {cls_acc:.4f}")
    print(f"  Precision: {cls_pre:.4f}")
    print(f"  Recall:    {cls_rec:.4f}")
    print(f"  F1-Score:  {cls_f1:.4f}")
    print(f"  MCC:       {cls_mcc:.4f}")
    print("="*30)

    #Print number of parameters in each model
    num_params_class = sum(p.numel() for p in classifier.parameters())
    num_params_reg   = sum(p.numel() for p in regressor.parameters())
    print(f"Model Complexity:")
    print(f"  Classifier parameters: {num_params_class:,}")
    print(f"  Regressor parameters:  {num_params_reg:,}")

    
    all_preds_np   = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    pipe_mse = mean_squared_error(all_targets_np, all_preds_np)
    pipe_mae = mean_absolute_error(all_targets_np, all_preds_np)
    pipe_r2  = r2_score(all_targets_np, all_preds_np)

    print(f"Final Pipeline Performance on Full Validation Set:")
    print(f"  MSE : {pipe_mse:.4f}")
    print(f"  MAE : {pipe_mae:.2f}")
    print(f"  R²  : {pipe_r2:.3f}")

    # Plot Pred vs Actual
    fig, ax = plt.subplots(figsize=(7, 7))
    x = all_targets_np
    y = all_preds_np
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    log_density = np.log(density)
    
    sc = ax.scatter(x, y, c=log_density, cmap="viridis", alpha=1, edgecolors="none", s=50)
    lims = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual People Count")
    ax.set_ylabel("Predicted People Count")
    ax.set_title("Predicted vs Actual (Full Pipeline)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_pred_vs_actual.png"), dpi=150)
    plt.close()


    #Loglog plot of pred vs actual with colormap based on density of points (darker = more points)
    fig, ax = plt.subplots(figsize=(7, 7))
    
    x = all_targets_np + 1
    y = all_preds_np + 1
    xy = np.vstack([np.log(x), np.log(y)])
    density = gaussian_kde(xy)(xy)
    log_density = np.log(density)  # compress the range so sparse points get more color variation

    
    sc = ax.scatter(x, y, c=log_density, cmap="viridis", alpha=1, edgecolors="none", s=50)
    
    lims = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual People Count + 1")   
    ax.set_ylabel("Predicted People Count + 1")
    ax.set_title("Predicted vs Actual (Log-Log Scale, Full Pipeline)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.legend(); ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_pred_vs_actual_loglog.png"), dpi=150)
    plt.close()

    # Plot Error Analysis
    errors     = all_preds_np - all_targets_np
    abs_errors = np.abs(errors)
    fig, axes  = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=30, edgecolor="black", alpha=0.7, color="#DD8452")
    axes[0].set_xlabel("Prediction Error (pred − actual)")
    axes[0].set_ylabel("Frequency"); axes[0].set_title("Pipeline Error Distribution")
    axes[0].axvline(0, color="red", linestyle="--")

    worst_idx = np.argsort(abs_errors)[-10:][::-1]
    axes[1].barh(range(len(worst_idx)), abs_errors[worst_idx], color="#C44E52", edgecolor="black")
    axes[1].set_xlabel("Absolute Error"); axes[1].set_ylabel("Sample Index")
    axes[1].set_title("Top 10 Worst Predictions")
    for i, idx in enumerate(worst_idx):
        axes[1].text(abs_errors[idx] + 0.1, i, f"actual={all_targets_np[idx]:.0f}, pred={all_preds_np[idx]:.1f}", va="center")
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pipeline_error_analysis.png"), dpi=150)
    plt.close()

    print(f"\nPipeline Error Statistics:")
    print(f"  Median absolute error: {np.median(abs_errors):.2f}")
    print(f"  90th percentile error: {np.percentile(abs_errors, 90):.2f}")
    print(f"  Within ±1 person:      {(abs_errors <= 1).mean()*100:.1f}%")
    print(f"  Within ±2 people:      {(abs_errors <= 2).mean()*100:.1f}%")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    # Force 'spawn' method which is safer for Windows/DirectML
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()