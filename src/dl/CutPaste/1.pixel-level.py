# ============================================================
# CUTPASTE — PIXEL-LEVEL ANOMALY LOCALIZATION
# ============================================================
# Estensione del modello CutPaste già addestrato.
# IMPORTANTE: Questo script NON esegue alcun training.
# Carica: results/cutpaste/cutpaste_best.pt e utilizza il 
# ResNet-18 già addestrato per estrarre feature spaziali da layer2.
# ============================================================

import os
import sys
import time
import random
from copy import deepcopy
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.linalg import pinvh

# ============================================================
# PATH
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from data_analysis.dataloader import get_dataset

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 128
NUM_WORKERS = 2
VAL_RATIO = 0.15
N_CLASSES = 3
GAUSSIAN_REGULARIZATION = 1e-5
PIXEL_THRESHOLD_PERCENTILE = 95

# ============================================================
# PIXEL-LEVEL CONFIGURATION
# ============================================================
FEATURE_LAYER = "layer2"
UPSAMPLE_MODE = "bilinear"
UPSAMPLE_ALIGN_CORNERS = False

# ============================================================
# OUTPUT & MODEL PATH
# ============================================================
BASE_OUT_DIR = os.path.join("results", "cutpaste")
PIXEL_OUT_DIR = os.path.join(BASE_OUT_DIR, "pixel_level")
MODEL_PATH = os.path.join(BASE_OUT_DIR, "cutpaste_best.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# RANDOM SEED & STYLE
# ============================================================
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.facecolor"] = "white"

# ============================================================
# MODEL
# ============================================================
class CutPasteModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)

        backbone.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, N_CLASSES)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        if FEATURE_LAYER == "layer1": return x
        
        x = backbone.layer2(x)
        if FEATURE_LAYER == "layer2": return x
        
        x = backbone.layer3(x)
        if FEATURE_LAYER == "layer3": return x
        
        x = backbone.layer4(x)
        return x

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
def load_trained_model(model_path: str) -> CutPasteModel:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nModello non trovato:\n{model_path}\n\n"
            f"Eseguire prima il training CutPaste."
        )

    print("\nLoading trained CutPaste model...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = CutPasteModel().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"  ✓ Loaded: {model_path}")
    print(f"  ✓ Feature dimension: {model.feature_dim}")
    return model

# ============================================================
# UTILS
# ============================================================
def create_train_validation_split(n_samples: int, val_ratio: float = VAL_RATIO, seed: int = SEED):
    rng = np.random.RandomState(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_val = max(1, min(n_samples - 1, int(n_samples * val_ratio)))
    return indices[n_val:], indices[:n_val]

def prepare_mask(mask) -> np.ndarray:
    if torch.is_tensor(mask): mask = mask.detach().cpu().numpy()
    mask = np.squeeze(np.asarray(mask))
    if mask.ndim != 2: raise ValueError(f"Shape mask errata: {mask.shape}")
    return (mask > 0).astype(np.uint8)

def prepare_image(image) -> np.ndarray:
    if torch.is_tensor(image): image = image.detach().cpu().numpy()
    image = np.squeeze(np.asarray(image))
    if image.ndim != 2: raise ValueError(f"Shape immagine errata: {image.shape}")
    return image

# ============================================================
# SPATIAL FEATURE EXTRACTION
# ============================================================
@torch.no_grad()
def extract_spatial_features(model: nn.Module, dataset, batch_size: int = BATCH_SIZE):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available()
    )
    model.eval()
    feature_batches, masks, patient_ids = [], [], []

    for batch in loader:
        images = batch["img"].float().to(DEVICE, non_blocking=True)
        spatial = model.forward_spatial(images)
        spatial = spatial.permute(0, 2, 3, 1).contiguous()
        feature_batches.append(spatial.cpu().numpy())
        
        if "patient_id" in batch:
            pids = batch["patient_id"]
            if isinstance(pids, torch.Tensor):
                pids = pids.cpu().numpy()
            patient_ids.extend(pids)
        else:
            patient_ids.extend(["unknown"] * len(images))

        if "mask" in batch:
            for mask in batch["mask"]:
                masks.append(prepare_mask(mask))

    features = np.concatenate(feature_batches, axis=0)
    return features, masks, np.array(patient_ids)


# ============================================================
# SPATIAL GAUSSIAN DENSITY ESTIMATOR
# ============================================================
class SpatialGaussianDensityEstimator:
    def __init__(self, regularization: float = GAUSSIAN_REGULARIZATION):
        self.regularization = regularization
        self.mean = None
        self.covariance = None
        self.precision = None
        self.feature_dim = None

    def fit(self, features: np.ndarray):
        n, h, w, d = features.shape
        self.feature_dim = d
        flat = features.reshape(-1, d).astype(np.float64)

        self.mean = np.mean(flat, axis=0)
        covariance = np.cov(flat, rowvar=False)

        if covariance.ndim == 0:
            covariance = np.array([[float(covariance)]])

        covariance = covariance + self.regularization * np.eye(covariance.shape[0])
        self.covariance = covariance
        self.precision = pinvh(covariance)

        print("\n  ✓ Spatial Gaussian fitted")
        print(f"    Samples: {flat.shape[0]:,}")
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        n, h, w, d = features.shape
        flat = features.reshape(-1, d).astype(np.float64)
        centered = flat - self.mean
        scores = np.einsum("ij,jk,ik->i", centered, self.precision, centered)
        return scores.reshape(n, h, w)

# ============================================================
# EVALUATION UTILS
# ============================================================
def upsample_anomaly_maps(anomaly_maps: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(anomaly_maps).float().unsqueeze(1)
    tensor = F.interpolate(
        tensor, size=target_size, mode=UPSAMPLE_MODE, align_corners=UPSAMPLE_ALIGN_CORNERS
    )
    return tensor.squeeze(1).numpy()

def compute_pixel_threshold(validation_maps: np.ndarray, validation_masks: List[np.ndarray], percentile: float = PIXEL_THRESHOLD_PERCENTILE) -> float:
    values = validation_maps.reshape(-1)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, percentile))

def compute_pixel_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict:
    y_true = np.asarray(y_true).astype(np.uint8)
    scores = np.asarray(scores).astype(np.float64)
    valid = np.isfinite(scores)
    y_true, scores = y_true[valid], scores[valid]

    y_pred = (scores >= threshold).astype(np.uint8)
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    balanced_accuracy = (sensitivity + specificity) / 2.0

    return {
        "pixel_auroc": float(auroc), "pixel_ap": float(ap), "pixel_f1": float(f1),
        "pixel_dice": float(dice), "pixel_iou": float(iou),
        "pixel_sensitivity": float(sensitivity), "pixel_specificity": float(specificity),
        "pixel_precision": float(precision), "pixel_balanced_accuracy": float(balanced_accuracy),
        "pixel_threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def compute_per_image_metrics(masks: List[np.ndarray], anomaly_maps: np.ndarray, threshold: float, labels: np.ndarray, patient_ids: np.ndarray) -> pd.DataFrame:
    rows = []
    for index in range(len(masks)):
        mask = prepare_mask(masks[index])
        score_map = anomaly_maps[index]
        prediction = (score_map >= threshold).astype(np.uint8)
        gt = (mask > 0).astype(np.uint8)

        intersection = np.logical_and(prediction == 1, gt == 1).sum()
        pred_area = (prediction == 1).sum()
        gt_area = (gt == 1).sum()
        union = np.logical_or(prediction == 1, gt == 1).sum()

        if pred_area == 0 and gt_area == 0:
            dice, iou = 1.0, 1.0
        else:
            dice = 2.0 * intersection / (pred_area + gt_area + 1e-8)
            iou = intersection / (union + 1e-8)

        rows.append({
            "index": index, 
            "patient_id": patient_ids[index], # <--- AGGIUNTO QUI
            "label": int(labels[index]), 
            "is_tumor": int(labels[index] == 1),
            "ground_truth_pixels": int(gt_area), 
            "predicted_anomaly_pixels": int(pred_area),
            "dice": float(dice), 
            "iou": float(iou)
        })
    return pd.DataFrame(rows)


def save_heatmap(image, mask, anomaly_map, threshold, label, index, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prediction = (anomaly_map >= threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(image, cmap="gray"); axes[1].imshow(mask, alpha=0.5); axes[1].set_title("Ground truth")
    im = axes[2].imshow(anomaly_map, cmap="inferno"); axes[2].set_title("Mahalanobis map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(image, cmap="gray"); axes[3].imshow(prediction, alpha=0.5); axes[3].set_title("Prediction")
    axes[4].imshow(image, cmap="gray"); axes[4].imshow(anomaly_map, cmap="inferno", alpha=0.55); axes[4].set_title("Overlay")
    
    for ax in axes: ax.axis("off")
    fig.suptitle(f"Pixel-level localization | index={index} | label={label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{index:05d}_label_{label}.png"), dpi=200, bbox_inches="tight")
    plt.close()

# ============================================================
# REPORTING SAVERS
# ============================================================
def save_pixel_metrics(metrics: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "pixel_level_metrics.csv"), index=False)

def save_pixel_report(metrics, train_shape, val_shape, test_shape, train_c, val_c, test_c, time_val, out_dir):
    path = os.path.join(out_dir, "pixel_level_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\nCUTPASTE — PIXEL-LEVEL ANOMALY LOCALIZATION\n" + "=" * 70 + "\n\n")
        f.write(f"Feature layer: {FEATURE_LAYER}\nTrain features: {train_shape}\n")
        f.write(f"Pixel AUROC: {metrics['pixel_auroc']:.6f}\nPixel Dice: {metrics['pixel_dice']:.6f}\n")
        f.write(f"Computation time: {time_val:.2f}s\n")

# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_pixel_level_experiment():
    total_start = time.time()
    os.makedirs(PIXEL_OUT_DIR, exist_ok=True)

    print("\n" + "=" * 70 + "\n CUTPASTE — PIXEL-LEVEL ANOMALY LOCALIZATION\n" + "=" * 70)
    print("\n[1/7] Loading datasets...")
    train_ds = get_dataset("brats", img_size=IMAGE_SIZE, mode="train")
    test_ds = get_dataset("brats", img_size=IMAGE_SIZE, mode="test")

    print("\n[3/7] Recreating train/validation split...")
    train_indices, val_indices = create_train_validation_split(len(train_ds))
    train_healthy_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_healthy_ds = torch.utils.data.Subset(train_ds, val_indices)

    model = load_trained_model(MODEL_PATH)

    print("\n[5/7] Extracting spatial features...")
    t0 = time.time()
    train_features, train_masks, _ = extract_spatial_features(model, train_healthy_ds)
    val_features, val_masks, _ = extract_spatial_features(model, val_healthy_ds)
    test_features, test_masks, test_pids = extract_spatial_features(model, test_ds)
    feature_extraction_time = time.time() - t0

    print("\n[6/7] Fitting spatial Gaussian...")
    gde = SpatialGaussianDensityEstimator(regularization=GAUSSIAN_REGULARIZATION)
    gde.fit(train_features)

    train_maps = upsample_anomaly_maps(gde.score(train_features), (IMAGE_SIZE, IMAGE_SIZE))
    val_maps = upsample_anomaly_maps(gde.score(val_features), (IMAGE_SIZE, IMAGE_SIZE))
    test_maps = upsample_anomaly_maps(gde.score(test_features), (IMAGE_SIZE, IMAGE_SIZE))

    pixel_threshold = compute_pixel_threshold(val_maps, val_masks, PIXEL_THRESHOLD_PERCENTILE)
    print(f"\n✓ Threshold P{PIXEL_THRESHOLD_PERCENTILE}: {pixel_threshold:.6f}")

    print("\n[7/7] Final pixel-level evaluation...")
    test_labels = np.asarray([test_ds[i]["label"] for i in range(len(test_ds))])
    
    y_pixel = np.concatenate([prepare_mask(m).reshape(-1) for m in test_masks], axis=0)
    pixel_scores = np.concatenate([s.reshape(-1) for s in test_maps], axis=0)

    metrics = compute_pixel_metrics(y_pixel, pixel_scores, pixel_threshold)

    print(f"\nPixel AUROC: {metrics['pixel_auroc']:.4f}")
    print(f"Pixel Dice:  {metrics['pixel_dice']:.4f}")

    print("\nComputing per-image metrics...")
    per_image_df = compute_per_image_metrics(test_masks, test_maps, pixel_threshold, test_labels, test_pids)
    per_image_df.to_csv(os.path.join(PIXEL_OUT_DIR, "per_image_pixel_metrics.csv"), index=False)

    heatmap_dir = os.path.join(PIXEL_OUT_DIR, "heatmaps")
    tumor_indices = np.where(test_labels == 1)[0]
    
    print("\nGenerating localization visualizations...")
    # SALVIAMO SOLO LE PRIME 10 IMMAGINI PER NON PERDERE TEMPO
    for index in tumor_indices[:10]: 
        sample = test_ds[index]
        save_heatmap(prepare_image(sample["img"]), prepare_mask(sample["mask"]), 
                     test_maps[index], pixel_threshold, 1, int(index), heatmap_dir)

    save_pixel_metrics(metrics, PIXEL_OUT_DIR)
    
    np.savez(os.path.join(PIXEL_OUT_DIR, "spatial_gaussian_density.npz"), 
             mean=gde.mean, covariance=gde.covariance, precision=gde.precision, 
             threshold=pixel_threshold, feature_layer=FEATURE_LAYER, image_size=IMAGE_SIZE)

    pd.DataFrame([{
        "seed": SEED, "image_size": IMAGE_SIZE, "feature_layer": FEATURE_LAYER,
        "pixel_threshold": pixel_threshold
    }]).to_csv(os.path.join(PIXEL_OUT_DIR, "pixel_level_config.csv"), index=False)

    save_pixel_report(metrics, train_features.shape, val_features.shape, test_features.shape, 
                      len(train_healthy_ds), len(val_healthy_ds), len(test_ds), feature_extraction_time, PIXEL_OUT_DIR)

    print(f"\nCOMPLETED in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    run_pixel_level_experiment()