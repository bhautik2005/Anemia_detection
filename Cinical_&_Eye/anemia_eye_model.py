"""
=============================================================
 AnemiaFusionNet — Eye Image Model (Module 1 of 3)
 Conjunctiva-based Anemia Detection using EfficientNetB3
=============================================================

DATASET FOLDER STRUCTURE EXPECTED:
------------------------------------
dataset/
├── Patient_001/
│   ├── 20200118_164733.jpg                    ← raw eye photo
│   ├── 20200118_164733_forniceal.png          ← forniceal region segment
│   ├── 20200118_164733_palpebral.png          ← palpebral region segment
│   ├── 20200118_164733_forniceal_palpebral.png← combined segment
│   └── ...
├── Patient_002/
│   └── ...
└── labels.csv   ← columns: Number, Hgb, Gender, Age, Note

labels.csv example:
  Number  Hgb   Gender  Age  Note
  1       12.2  M       29
  2       8.0   F       36

=============================================================
"""

# ── 0. Imports ────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# ── 1. Configuration ──────────────────────────────────────
class Config:
    # --- Paths (change these to match your folder) ---
    DATASET_DIR   = "detaset/India"          # root folder with patient subfolders
    LABELS_CSV    = "detaset/labels.xlsx"   # Excel/CSV file with Hgb values
    OUTPUT_DIR    = "outputs"               # where model & plots are saved

    # --- Which image type to use ---
    # Options: "raw", "forniceal", "palpebral", "forniceal_palpebral"
    IMAGE_TYPE    = "palpebral"        # palpebral works best for anemia

    # --- Anemia threshold (g/dL) ---
    # WHO definition: <12 for women, <13 for men. Using 12 as general cutoff.
    ANEMIA_THRESHOLD = 12.0            # Hgb < 12 → Anemic (1), else Non-Anemic (0)

    # --- Model hyperparameters ---
    IMAGE_SIZE    = 224                # EfficientNetB3 input size
    BATCH_SIZE    = 16
    EPOCHS        = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY  = 1e-5
    DROPOUT_RATE  = 0.4

    # --- Train/Val/Test split ---
    VAL_SIZE      = 0.15
    TEST_SIZE     = 0.15
    RANDOM_SEED   = 42

    # --- Device ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(f"Using device: {cfg.DEVICE}")


# ── 2. Label Loading ──────────────────────────────────────
def load_labels(csv_path: str, threshold: float) -> pd.DataFrame:
    """
    Reads the label workbook or CSV and creates binary anemia labels.
    Expected columns: Number, Hgb, Gender, Age
    """
    candidates = []
    if csv_path:
        candidates.append(Path(csv_path))
        candidates.append(Path("detaset") / Path(csv_path).name)
        candidates.append(Path("dataset") / Path(csv_path).name)

    # Also try common default locations if the provided path does not exist
    candidates.extend([
        Path("detaset/labels.xlsx"),
        Path("detaset/labels.csv"),
        Path("dataset/labels.xlsx"),
        Path("dataset/labels.csv"),
    ])

    resolved_path = None
    for candidate in candidates:
        if candidate.exists():
            resolved_path = candidate
            break

    if resolved_path is None:
        raise FileNotFoundError(f"Could not find labels file. Checked: {', '.join(str(c) for c in candidates)}")

    print(f"Using labels file: {resolved_path}")

    if resolved_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(resolved_path)
    else:
        df = pd.read_csv(resolved_path)

    # Normalize column names (strip spaces, lowercase)
    df.columns = df.columns.str.strip().str.lower()

    if "number" not in df.columns or "hgb" not in df.columns:
        raise ValueError(f"Expected columns 'Number' and 'Hgb' in labels file, found: {list(df.columns)}")

    df["number"] = pd.to_numeric(df["number"], errors="coerce")
    df["hgb"] = pd.to_numeric(df["hgb"], errors="coerce")
    df = df.dropna(subset=["number", "hgb"]).reset_index(drop=True)

    # Create binary label: 1 = Anemic, 0 = Non-Anemic
    df["label"] = (df["hgb"] < threshold).astype(int)

    print(f"\nDataset label distribution:")
    print(df["label"].value_counts().rename({0: "Non-Anemic", 1: "Anemic"}))
    print(f"Total samples: {len(df)}")

    return df


# ── 3. Image Path Resolver ────────────────────────────────
def get_image_path(dataset_dir: str, patient_number: int, image_type: str) -> str:
    """
    Finds the correct image file for a patient.
    Tries multiple common naming patterns.
    """
    folder_candidates = [
        f"Patient_{patient_number:03d}",
        f"Patient_{patient_number}",
        str(patient_number),
        f"{patient_number:03d}",
    ]

    # Filename suffixes for each image type
    suffix_map = {
        "raw":                   [".jpg", ".jpeg", ".png"],
        "forniceal":             ["_forniceal.png", "_forniceal.jpg"],
        "palpebral":             ["_palpebral.png", "_palpebral.jpg"],
        "forniceal_palpebral":   ["_forniceal_palpebral.png", "_forniceal_palpebral.jpg"],
    }

    for folder_name in folder_candidates:
        patient_folder = Path(dataset_dir) / folder_name
        if not patient_folder.exists():
            continue

        suffixes = suffix_map.get(image_type, [".jpg"])

        for file in sorted(patient_folder.iterdir()):
            fname = file.name.lower()
            for suffix in suffixes:
                if fname.endswith(suffix.lower()):
                    # For "raw" type, make sure it's not a segmented image
                    if image_type == "raw":
                        if not any(s in fname for s in ["_forniceal", "_palpebral"]):
                            return str(file)
                    else:
                        return str(file)

    return None


# ── 4. Dataset Class ──────────────────────────────────────
class ConjunctivaDataset(Dataset):
    """
    PyTorch Dataset for conjunctiva eye images.
    Handles loading, augmentation, and label retrieval.
    """

    def __init__(self, records: list, transform=None):
        """
        records: list of dicts with keys 'image_path', 'label', 'hgb'
        """
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        img_path = record["image_path"]

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            image = Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), color=(200, 150, 150))

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(record["label"], dtype=torch.long)
        hgb   = torch.tensor(record["hgb"],   dtype=torch.float32)

        return image, label, hgb


# ── 5. Data Transforms ────────────────────────────────────
def get_transforms(image_size: int):
    """
    Returns train and validation/test image transforms.

    Train: augmentation (flip, rotate, color jitter) to reduce overfitting
    Val/Test: only resize + normalize (no augmentation)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,   # simulate different lighting
            contrast=0.3,
            saturation=0.3,   # key for pallor detection
            hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


# ── 6. Model Architecture ─────────────────────────────────
class AnemiaEyeModel(nn.Module):
    """
    EfficientNetB3 backbone with custom classification head.

    Why EfficientNetB3?
    - Good balance of accuracy vs. speed
    - Pre-trained on ImageNet (transfer learning)
    - Handles subtle color/texture differences well (important for pallor)

    Architecture:
        EfficientNetB3 (frozen early layers) →
        Dropout →
        FC 1536 → 512 →
        Dropout →
        FC 512 → 2 (Anemic / Non-Anemic)
    """

    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.4):
        super().__init__()

        # Load pretrained EfficientNetB3, but fall back to random initialization if weights are unavailable.
        try:
            self.backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"Warning: Could not load pretrained EfficientNet weights ({e}). Falling back to random initialization.")
            self.backbone = efficientnet_b3(weights=None)

        # Freeze early layers (keep color/texture features from ImageNet)
        # Only unfreeze last 3 blocks for fine-tuning
        layers = list(self.backbone.features.children())
        for layer in layers[:-3]:
            for param in layer.parameters():
                param.requires_grad = False

        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features  # 1536 for B3

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

        # Store feature extractor output for later fusion use
        self._feature_dim = 512

    def forward(self, x):
        return self.backbone(x)

    def extract_features(self, x):
        """
        Returns 512-dim feature vector (for fusion with clinical + geo modules).
        Used in AnemiaFusionNet — do NOT change output size.
        """
        # Run through backbone up to classifier
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)

        # Run through first part of classifier (up to the last linear layer)
        feat = self.backbone.classifier[0](features)  # Dropout
        feat = self.backbone.classifier[1](feat)       # Linear → 512
        feat = self.backbone.classifier[2](feat)       # ReLU
        feat = self.backbone.classifier[3](feat)       # BatchNorm
        feat = self.backbone.classifier[4](feat)       # Dropout

        return feat  # 512-dim vector


# ── 7. Class Weights (handle imbalanced data) ─────────────
def compute_class_weights(labels: list) -> torch.Tensor:
    """
    Computes inverse frequency weights for imbalanced datasets.
    If 70% non-anemic and 30% anemic, anemic gets higher weight.
    """
    label_array = np.array(labels)
    class_counts = np.bincount(label_array)
    total = len(label_array)
    weights = total / (len(class_counts) * class_counts)
    print(f"Class weights → Non-Anemic: {weights[0]:.3f}, Anemic: {weights[1]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(cfg.DEVICE)


# ── 8. Training Loop ──────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── 9. Validation Loop ────────────────────────────────────
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += images.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    f1  = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return total_loss / total, correct / total, f1, auc


# ── 10. Plot Training History ─────────────────────────────
def plot_training_history(history: dict, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train", color="#2a78d6")
    axes[0].plot(history["val_loss"],   label="Val",   color="#e34948")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train", color="#2a78d6")
    axes[1].plot(history["val_acc"],   label="Val",   color="#e34948")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["val_f1"],  label="F1",  color="#1baf7a")
    axes[2].plot(history["val_auc"], label="AUC", color="#4a3aa7")
    axes[2].set_title("Val F1 & AUC-ROC")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training history saved → {save_path}")


# ── 11. Plot Confusion Matrix ─────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Anemic", "Anemic"],
        yticklabels=["Non-Anemic", "Anemic"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ── 12. Plot ROC Curve ────────────────────────────────────
def plot_roc_curve(y_true, y_probs, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#2a78d6", label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved → {save_path}")


# ── 13. Main Training Pipeline ────────────────────────────
def main():
    print("=" * 55)
    print(" AnemiaFusionNet — Eye Image Model Training")
    print("=" * 55)

    # ---- Step 1: Load labels ----
    df = load_labels(cfg.LABELS_CSV, cfg.ANEMIA_THRESHOLD)

    # ---- Step 2: Build records list ----
    records = []
    missing = 0
    for _, row in df.iterrows():
        img_path = get_image_path(
            cfg.DATASET_DIR,
            int(row["number"]),
            cfg.IMAGE_TYPE
        )
        if img_path is None:
            missing += 1
            continue
        records.append({
            "image_path": img_path,
            "label":      int(row["label"]),
            "hgb":        float(row["hgb"]),
            "patient_id": int(row["number"]),
        })

    print(f"\nFound {len(records)} images, {missing} missing/skipped")
    if len(records) == 0:
        print("ERROR: No images found! Check DATASET_DIR and folder structure.")
        return

    # ---- Step 3: Train / Val / Test split ----
    labels_list = [r["label"] for r in records]

    train_records, temp_records = train_test_split(
        records, test_size=cfg.VAL_SIZE + cfg.TEST_SIZE,
        stratify=labels_list, random_state=cfg.RANDOM_SEED
    )
    temp_labels = [r["label"] for r in temp_records]
    val_records, test_records = train_test_split(
        temp_records,
        test_size=cfg.TEST_SIZE / (cfg.VAL_SIZE + cfg.TEST_SIZE),
        stratify=temp_labels, random_state=cfg.RANDOM_SEED
    )

    print(f"\nSplit: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")

    # ---- Step 4: Datasets & DataLoaders ----
    train_tf, val_tf = get_transforms(cfg.IMAGE_SIZE)

    train_ds = ConjunctivaDataset(train_records, transform=train_tf)
    val_ds   = ConjunctivaDataset(val_records,   transform=val_tf)
    test_ds  = ConjunctivaDataset(test_records,  transform=val_tf)

    num_workers = 0 if os.name == "nt" else 2
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,  num_workers=num_workers, pin_memory=(cfg.DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(cfg.DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=(cfg.DEVICE == "cuda"))

    # ---- Step 5: Model ----
    model = AnemiaEyeModel(num_classes=2, dropout_rate=cfg.DROPOUT_RATE)
    model = model.to(cfg.DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EfficientNetB3  |  Trainable params: {trainable:,} / {total:,}")

    # ---- Step 6: Loss, Optimizer, Scheduler ----
    class_weights = compute_class_weights([r["label"] for r in train_records])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # Cosine annealing: smoothly reduces LR to near-zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS, eta_min=1e-6
    )

    # ---- Step 7: Training loop ----
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc": [],
        "val_f1":     [], "val_auc": []
    }
    best_val_f1    = 0.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, "best_eye_model.pth")
    patience       = 8   # early stopping patience
    no_improve     = 0

    print(f"\nStarting training for {cfg.EPOCHS} epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'F1':>6} | {'AUC':>6}")
    print("-" * 70)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE)
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>9.4f} | {val_loss:>8.4f} | {val_acc:>7.4f} | {val_f1:>6.4f} | {val_auc:>6.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1":     val_f1,
                "val_auc":    val_auc,
                "config":     vars(cfg),
            }, best_model_path)
            print(f"         ✓ Best model saved (F1={val_f1:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # ---- Step 8: Test evaluation ----
    print("\n" + "=" * 55)
    print(" Final Test Evaluation")
    print("=" * 55)

    # Load best model
    checkpoint = torch.load(best_model_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, test_acc, test_f1, test_auc = validate(model, test_loader, criterion, cfg.DEVICE)

    # Get predictions for detailed report
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(cfg.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print(f"\nTest Accuracy : {test_acc:.4f}")
    print(f"Test F1 Score : {test_f1:.4f}")
    print(f"Test AUC-ROC  : {test_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Non-Anemic", "Anemic"]))

    # ---- Step 9: Save plots ----
    plot_training_history(history, os.path.join(cfg.OUTPUT_DIR, "training_history.png"))
    plot_confusion_matrix(all_labels, all_preds, os.path.join(cfg.OUTPUT_DIR, "confusion_matrix.png"))
    plot_roc_curve(all_labels, all_probs, os.path.join(cfg.OUTPUT_DIR, "roc_curve.png"))

    print(f"\nAll outputs saved to: {cfg.OUTPUT_DIR}/")
    print(f"Best model: {best_model_path}")
    print("\nDone!")


# ── 14. Feature Extraction (for Fusion) ───────────────────
def extract_features_for_fusion(image_paths: list, model_path: str) -> np.ndarray:
    """
    Use this function AFTER training to extract 512-dim vectors.
    These vectors feed into the Transformer Fusion module (Module 4).

    Usage:
        features = extract_features_for_fusion(image_paths, "outputs/best_eye_model.pth")
        # features.shape → (N, 512)
    """
    model = AnemiaEyeModel(num_classes=2)
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.DEVICE)
    model.eval()

    _, val_tf = get_transforms(cfg.IMAGE_SIZE)
    all_features = []

    with torch.no_grad():
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = val_tf(img).unsqueeze(0).to(cfg.DEVICE)
                feat = model.extract_features(img)
                all_features.append(feat.cpu().numpy())
            except Exception as e:
                print(f"Warning: {path}: {e}")
                all_features.append(np.zeros((1, 512)))

    return np.vstack(all_features)


# ── 15. Single Image Prediction ───────────────────────────
def predict_single_image(image_path: str, model_path: str) -> dict:
    """
    Predict anemia for a single image.
    Returns dict with label, confidence, and feature vector.

    Usage:
        result = predict_single_image("patient_eye.jpg", "outputs/best_eye_model.pth")
        print(result["prediction"])   # "Anemic" or "Non-Anemic"
        print(result["confidence"])   # e.g. 0.87
    """
    model = AnemiaEyeModel(num_classes=2)
    checkpoint = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.DEVICE)
    model.eval()

    _, val_tf = get_transforms(cfg.IMAGE_SIZE)

    img  = Image.open(image_path).convert("RGB")
    inp  = val_tf(img).unsqueeze(0).to(cfg.DEVICE)

    with torch.no_grad():
        output = model(inp)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()
        feat   = model.extract_features(inp)

    return {
        "prediction":   "Anemic" if pred == 1 else "Non-Anemic",
        "confidence":   round(probs[pred].item(), 4),
        "prob_anemic":  round(probs[1].item(), 4),
        "prob_normal":  round(probs[0].item(), 4),
        "feature_vector": feat.cpu().numpy(),  # 512-dim, use for fusion
    }


# ── Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    main()