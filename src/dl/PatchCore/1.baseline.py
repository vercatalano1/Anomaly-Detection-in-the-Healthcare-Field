# ============================================================
# PATCHCORE — TRANSFER LEARNING ANOMALY DETECTION
# ============================================================
#
# Based on:
# Roth et al., "Towards Total Recall in Industrial Anomaly 
# Detection" (CVPR 2022).
#
# Protocollo:
# Nessun addestramento viene eseguito (Transfer Learning puro).
# Le immagini in scala di grigi vengono adattate a 3 canali.
# Viene estratta una gerarchia di feature (layer1, layer2)
# da una ResNet-18 pre-addestrata su ImageNet.
# Le feature dei pazienti sani formano una "Memory Bank",
# compressa tramite Coreset Subsampling (Random Projection).
# Le anomalie in fase di test vengono identificate calcolando
# la L2-distance (K-Nearest Neighbors) tra la patch del test
# e le patch sane nella Memory Bank.
# ============================================================

import os
import sys
import time
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    balanced_accuracy_score,
)

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

# Core PatchCore settings
SUBSAMPLING_RATIO = 0.05  # Percentuale di feature sane da tenere in memoria
BLUR_SIGMA = 2.0          # Smoothing per rendere le mappe morbide
THRESHOLD_PERCENTILE = 95 # Soglia P95

# Output
OUT_DIR = os.path.join("results", "patchcore")

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# MODEL: PATCHCORE BACKBONE
# ============================================================
class PatchCoreBackbone(nn.Module):
    """
    ResNet-18 ImageNet pretrained.
    Estrae feature spaziali (layer1, layer2).
    Poiché l'input originale era a 3 canali e noi abbiamo grayscale,
    l'input verrà duplicato 3 volte nel forward.
    """
    def __init__(self):
        super().__init__()
        # Prende i pesi di default addestrati su milioni di immagini a colori
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = backbone
        self.backbone.eval()
        
        # Svuotiamo i layer finali, non ci servono
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # Hooks per estrarre le feature intermedie
        self.features = []
        def hook_fn(module, input, output):
            self.features.append(output)
        
        self.backbone.layer1.register_forward_hook(hook_fn)
        self.backbone.layer2.register_forward_hook(hook_fn)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.features = []
        # DUPLICAZIONE CANALI: 1 -> 3 canali per adattarsi a ImageNet
        x = x.repeat(1, 3, 1, 1)
        _ = self.backbone(x)
        
        # Layer 1: [B, 64, 16, 16] per input 64x64
        # Layer 2: [B, 128, 8, 8] per input 64x64
        feat_1 = self.features[0]
        feat_2 = self.features[1]
        
        # Patch Pooling (Aumenta il campo recettivo locale)
        feat_1 = F.avg_pool2d(feat_1, 3, 1, 1)
        feat_2 = F.avg_pool2d(feat_2, 3, 1, 1)

        # Upsampling del Layer 2 per matchare il Layer 1
        feat_2 = F.interpolate(feat_2, size=feat_1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenazione: [B, 192, 16, 16]
        features = torch.cat([feat_1, feat_2], dim=1)
        return features

# ============================================================
# FEATURE EXTRACTION & MEMORY BANK
# ============================================================
@torch.no_grad()
def build_memory_bank(model: nn.Module, dataset) -> torch.Tensor:
    """Estrae e campiona casualmente le feature per costruire la Memory Bank"""
    print("\n  Extracting features for Memory Bank...")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    
    bank_features = []
    for batch in loader:
        images = batch["img"].float().to(DEVICE)
        features = model(images) # [B, C, H, W]
        # Trasponi a [B, H, W, C] e appiattisci
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(b * h * w, c)
        bank_features.append(features.cpu())
        
    bank_features = torch.cat(bank_features, dim=0)
    print(f"  Total healthy patches extracted: {bank_features.shape[0]:,}")
    
    # Random Subsampling (Coreset approssimato)
    n_keep = int(bank_features.shape[0] * SUBSAMPLING_RATIO)
    indices = torch.randperm(bank_features.shape[0])[:n_keep]
    memory_bank = bank_features[indices].to(DEVICE) # Spostiamo su GPU per velocità
    
    print(f"  Memory Bank size (after {SUBSAMPLING_RATIO*100}% subsampling): {memory_bank.shape[0]:,}")
    return memory_bank

@torch.no_grad()
def evaluate_patchcore(model: nn.Module, memory_bank: torch.Tensor, dataset) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """Valuta il dataset calcolando la distanza KNN con la Memory Bank"""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    
    image_scores = []
    anomaly_maps = []
    masks = []
    labels = []

    for batch in loader:
        images = batch["img"].float().to(DEVICE)
        labels.extend(batch["label"].numpy())
        if "mask" in batch:
            for mask in batch["mask"]: masks.append(prepare_mask(mask))

        features = model(images) # [B, C, H, W]
        b, c, h, w = features.shape
        features = features.permute(0, 2, 3, 1).reshape(b * h * w, c) # [B*H*W, C]
        
        # Distanza euclidea vettorizzata a blocchi
        distances = torch.cdist(features, memory_bank, p=2.0) # [B*H*W, K]
        min_distances, _ = torch.min(distances, dim=1) # [B*H*W]
        
        # Rimodella a mappa spaziale
        score_map = min_distances.reshape(b, 1, h, w)
        
        # Upsampling alla dimensione dell'immagine originale (64x64)
        score_map = F.interpolate(score_map, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        score_map = score_map.squeeze(1).cpu().numpy()
        
        # Gaussian smoothing e Image Score
        for i in range(b):
            smoothed_map = gaussian_filter(score_map[i], sigma=BLUR_SIGMA)
            anomaly_maps.append(smoothed_map)
            image_scores.append(np.max(smoothed_map)) # Max patch distance

    return np.array(image_scores), np.array(anomaly_maps), masks, np.array(labels)

# ============================================================
# EVALUATION METRICS
# ============================================================
def compute_metrics(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    auroc = roc_auc_score(y_true, scores)
    ap = average_precision_score(y_true, scores)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8)
    bacc = balanced_accuracy_score(y_true, y_pred)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8) if tp+fp+fn > 0 else 0.0

    return {
        "auroc": float(auroc), "ap": float(ap), "f1": float(f1),
        "dice": float(dice), "sensitivity": float(sensitivity), 
        "specificity": float(specificity), "precision": float(precision), 
        "bacc": float(bacc), "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "cm": cm
    }

# ============================================================
# VISUALIZATION
# ============================================================
def save_heatmap(image, mask, anomaly_map, threshold, label, index, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    prediction = (anomaly_map >= threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(image, cmap="gray"); axes[1].imshow(mask, alpha=0.5); axes[1].set_title("Ground truth")
    im = axes[2].imshow(anomaly_map, cmap="inferno"); axes[2].set_title("PatchCore KNN Map")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(image, cmap="gray"); axes[3].imshow(prediction, alpha=0.5); axes[3].set_title("Prediction")
    axes[4].imshow(image, cmap="gray"); axes[4].imshow(anomaly_map, cmap="inferno", alpha=0.55); axes[4].set_title("Overlay")
    
    for ax in axes: ax.axis("off")
    fig.suptitle(f"PatchCore Localization | index={index} | label={label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{index:05d}_label_{label}.png"), dpi=200, bbox_inches="tight")
    plt.close()




# ============================================================
# IMAGE-LEVEL RESULTS FIGURE
# ============================================================
def plot_image_level_results(y_test, scores, threshold, auroc, ap, cm, out_dir):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_test, scores)
    precision_curve, recall, _ = precision_recall_curve(y_test, scores)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # (A) SCORE DISTRIBUTION
    ax = axes[0, 0]
    healthy = scores[y_test == 0]
    tumor = scores[y_test == 1]
    if len(healthy) > 1: sns.kdeplot(healthy, fill=True, color="#2ca02c", alpha=0.4, label="Healthy", ax=ax, linewidth=2)
    if len(tumor) > 1: sns.kdeplot(tumor, fill=True, color="#d62728", alpha=0.4, label="Tumor", ax=ax, linewidth=2)
    ax.axvline(threshold, linestyle="--", color="black", linewidth=2, label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("(A) Image-level Anomaly Scores", fontweight="bold")
    ax.legend()
    
    # (B) CONFUSION MATRIX
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["Healthy", "Tumor"], yticklabels=["Healthy", "Tumor"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("(B) Confusion Matrix", fontweight="bold")
    
    # (C) ROC
    ax = axes[1, 0]
    ax.plot(fpr, tpr, linewidth=2.5, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("(C) ROC Curve", fontweight="bold")
    ax.legend(loc="lower right")
    
    # (D) PR
    ax = axes[1, 1]
    baseline = np.mean(y_test)
    ax.plot(recall, precision_curve, linewidth=2.5, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, linestyle="--", color="gray", alpha=0.6, label=f"Baseline = {baseline:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(D) Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    path = os.path.join(out_dir, "image_level_results.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


# ============================================================
# REPORT
# ============================================================
def save_report(metrics_img, metrics_pix, bank_size, exec_time, out_dir):
    path = os.path.join(out_dir, "patchcore_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("PATCHCORE — TRANSFER LEARNING ANOMALY DETECTION\n")
        f.write("======================================================================\n\n")
        f.write(f"Backbone: ResNet-18 (ImageNet Pretrained)\n")
        f.write(f"Layers extracted: layer1, layer2\n")
        f.write(f"Memory Bank Size: {bank_size} patches (Subsampling: {SUBSAMPLING_RATIO*100}%)\n")
        f.write(f"Gaussian Blur Sigma: {BLUR_SIGMA}\n")
        f.write(f"Execution time: {exec_time:.2f}s\n\n")
        f.write("IMAGE-LEVEL RESULTS\n----------------------------------------------------------------------\n")
        f.write(f"AUROC:                 {metrics_img['auroc']:.4f}\n")
        f.write(f"Average Precision:     {metrics_img['ap']:.4f}\n")
        f.write(f"Sensitivity:           {metrics_img['sensitivity']:.4f}\n")
        f.write(f"Specificity:           {metrics_img['specificity']:.4f}\n")
        f.write(f"Threshold:             {metrics_img['threshold']:.6f}\n\n")
        f.write("PIXEL-LEVEL RESULTS\n----------------------------------------------------------------------\n")
        f.write(f"Pixel AUROC:           {metrics_pix['auroc']:.4f}\n")
        f.write(f"Pixel Dice:            {metrics_pix['dice']:.4f}\n")
        f.write(f"Pixel Sensitivity:     {metrics_pix['sensitivity']:.4f}\n")
        f.write(f"Pixel Specificity:     {metrics_pix['specificity']:.4f}\n")
        f.write(f"Pixel Threshold:       {metrics_pix['threshold']:.6f}\n")
    print(f"✓ Saved: {path}")

# ============================================================
# MAIN
# ============================================================
def run_patchcore_experiment():
    total_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 70 + "\n PATCHCORE — TRANSFER LEARNING ANOMALY DETECTION\n" + "=" * 70)
    
    print("\n[1/5] Loading datasets...")
    train_ds = get_dataset("brats", img_size=IMAGE_SIZE, mode="train")
    test_ds = get_dataset("brats", img_size=IMAGE_SIZE, mode="test")

    print("\n[2/5] Creating train/validation split...")
    train_indices, val_indices = create_train_validation_split(len(train_ds))
    train_healthy_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_healthy_ds = torch.utils.data.Subset(train_ds, val_indices)

    print("\n[3/5] Initializing Pre-trained Backbone...")
    model = PatchCoreBackbone().to(DEVICE)

    # MEMORY BANK
    print("\n[4/5] Building Healthy Memory Bank...")
    t0 = time.time()
    memory_bank = build_memory_bank(model, train_healthy_ds)

    # VALIDATION (Per trovare le soglie)
    print("\n[5/5] Scoring datasets via K-Nearest Neighbors...")
    print("  Evaluating Validation Set (to find P95 thresholds)...")
    val_img_scores, val_maps, val_masks, _ = evaluate_patchcore(model, memory_bank, val_healthy_ds)
    
    img_threshold = float(np.percentile(val_img_scores, THRESHOLD_PERCENTILE))
    pix_threshold = float(np.percentile(val_maps.flatten(), THRESHOLD_PERCENTILE))
    print(f"  ✓ Image P95 Threshold: {img_threshold:.6f}")
    print(f"  ✓ Pixel P95 Threshold: {pix_threshold:.6f}")

    # TEST
    print("  Evaluating Test Set...")
    test_img_scores, test_maps, test_masks, test_labels = evaluate_patchcore(model, memory_bank, test_ds)
    exec_time = time.time() - t0

    # METRICS
    print("\nCalculating metrics...")
    metrics_img = compute_metrics(test_labels, test_img_scores, img_threshold)
    
    y_pixel = np.concatenate([m.flatten() for m in test_masks], axis=0)
    pixel_scores = np.concatenate([s.flatten() for s in test_maps], axis=0)
    metrics_pix = compute_metrics(y_pixel, pixel_scores, pix_threshold)

    print("\n" + "=" * 70 + "\n PATCHCORE TEST RESULTS\n" + "=" * 70)
    print(f"IMAGE AUROC: {metrics_img['auroc']:.4f}")
    print(f"IMAGE SENS:  {metrics_img['sensitivity']:.4f}\n")
    print(f"PIXEL AUROC: {metrics_pix['auroc']:.4f}")
    print(f"PIXEL DICE:  {metrics_pix['dice']:.4f}")

    # SAVING
    pd.DataFrame([metrics_img]).to_csv(os.path.join(OUT_DIR, "patchcore_image_metrics.csv"), index=False)
    pd.DataFrame([metrics_pix]).to_csv(os.path.join(OUT_DIR, "patchcore_pixel_metrics.csv"), index=False)
    
    save_report(metrics_img, metrics_pix, memory_bank.shape[0], exec_time, OUT_DIR)

    print("\nGenerating image-level plots...")
    plot_image_level_results(
        test_labels, test_img_scores, img_threshold, 
        metrics_img['auroc'], metrics_img['ap'], metrics_img['cm'], OUT_DIR
    )

    heatmap_dir = os.path.join(OUT_DIR, "heatmaps")
    tumor_indices = np.where(test_labels == 1)[0]
    
    print("\nGenerating localization visualizations...")
    # Salviamo solo 10 esempi tumorali per evitare di inondare il disco
    for index in tumor_indices[:10]:
        sample = test_ds[int(index)]
        save_heatmap(prepare_image(sample["img"]), prepare_mask(sample["mask"]), 
                     test_maps[index], pix_threshold, 1, int(index), heatmap_dir)

    print(f"\nCOMPLETED in {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    run_patchcore_experiment()