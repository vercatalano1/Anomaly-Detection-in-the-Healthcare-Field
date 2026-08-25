# ============================================================
# CNN AUTOENCODER — DEEP UNSUPERVISED BASELINE
# ============================================================
#
# Tesi:
#   Studio e analisi di tecniche di anomaly detection
#   per tumori cerebrali.
#
# Dataset:
#   BraTS2021 / MedIAnomaly
#
# Paradigma:
#
#   TRAIN:
#       esclusivamente immagini HEALTHY
#
#   VALIDATION:
#       esclusivamente immagini HEALTHY
#       utilizzata per:
#           - early stopping
#           - threshold selection
#
#   TEST:
#       HEALTHY + TUMOR
#       utilizzato esclusivamente per la valutazione finale
#
# Modello:
#   Convolutional Autoencoder
#
# Image-level:
#   anomaly_score = reconstruction MSE
#
# Pixel-level:
#   anomaly_map = |input - reconstruction|
#
# Ground truth pixel-level:
#   BraTS2021/test/annotation/
#
# Importante:
#   Le tumor labels e le segmentation masks NON vengono
#   utilizzate durante il training.
#
# ============================================================


import os
import re
import sys
import time
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    balanced_accuracy_score,
)

# 1. Trova il percorso assoluto della cartella 'ml'
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Trova la cartella "padre" (la root del tuo progetto)
parent_dir = os.path.dirname(current_dir)

# 3. Aggiunge la cartella padre ai percorsi di sistema di Python
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_analysis.dataloader import get_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 128

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

LATENT_DIM = 128

VAL_RATIO = 0.15

PATIENCE = 10

MIN_DELTA = 1e-5

NUM_WORKERS = 0

OUT_DIR = "results/cnn_autoencoder"

DATA_ROOT = "BraTS2021"

TEST_ROOT = os.path.join(
    DATA_ROOT,
    "test"
)

TUMOR_DIR = os.path.join(
    TEST_ROOT,
    "tumor"
)

ANNOTATION_DIR = os.path.join(
    TEST_ROOT,
    "annotation"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = SEED) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# PLOT STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)

plt.rcParams["figure.facecolor"] = "white"


# ============================================================
# DATA EXTRACTION
# ============================================================

def extract_features(
    dataset,
    batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae immagini e label dal dataset.

    Output:

        X:
            [N, 1, 64, 64]

        y:
            [N]

    Le immagini vengono mantenute in formato
    spaziale perché il CNN Autoencoder lavora
    direttamente sulla struttura 2D.
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    X_list = []

    y_list = []

    for batch in loader:

        imgs = batch["img"].cpu().numpy()

        labels = batch["label"].cpu().numpy()

        X_list.append(imgs)

        y_list.append(labels)

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    return X.astype(
        np.float32
    ), y.astype(
        np.int64
    )


# ============================================================
# PATIENT IDS
# ============================================================

def extract_patient_ids(
    dataset
) -> np.ndarray:
    """
    Estrae patient_id mantenendo l'ordine del dataset.
    """

    patient_ids = []

    for i in range(
        len(dataset)
    ):

        sample = dataset[i]

        patient_ids.append(
            sample["patient_id"]
        )

    return np.asarray(
        patient_ids
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    n_samples: int,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide il training healthy in:

        TRAIN healthy
        VALIDATION healthy

    Nessuna informazione relativa al test
    viene utilizzata.
    """

    rng = np.random.RandomState(
        seed
    )

    indices = np.arange(
        n_samples
    )

    rng.shuffle(
        indices
    )

    n_val = int(
        n_samples * val_ratio
    )

    val_indices = indices[
        :n_val
    ]

    train_indices = indices[
        n_val:
    ]

    return (
        train_indices,
        val_indices
    )


# ============================================================
# CNN AUTOENCODER
# ============================================================

class Encoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim: int = LATENT_DIM
    ):

        super().__init__()

        self.features = nn.Sequential(

            # 64 x 64
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            # 32 x 32
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 16 x 16
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 8 x 8
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            # 4 x 4
        )

        self.fc = nn.Linear(
            256 * 4 * 4,
            latent_dim
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        x = x.flatten(
            start_dim=1
        )

        z = self.fc(
            x
        )

        return z


class Decoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim: int = LATENT_DIM
    ):

        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            256 * 4 * 4
        )

        self.features = nn.Sequential(

            # 4 x 4
            nn.ConvTranspose2d(
                256,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            # 8 x 8
            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            # 16 x 16
            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            # 32 x 32
            nn.ConvTranspose2d(
                32,
                1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )

    def forward(
        self,
        z
    ):

        x = self.fc(
            z
        )

        x = x.view(
            -1,
            256,
            4,
            4
        )

        x = self.features(
            x
        )

        return x


class ConvAutoencoder(
    nn.Module
):

    def __init__(
        self,
        latent_dim: int = LATENT_DIM
    ):

        super().__init__()

        self.encoder = Encoder(
            latent_dim
        )

        self.decoder = Decoder(
            latent_dim
        )

    def forward(
        self,
        x
    ):

        z = self.encoder(
            x
        )

        reconstruction = self.decoder(
            z
        )

        return reconstruction, z


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion
) -> float:

    model.train()

    running_loss = 0.0

    n_samples = 0

    for batch in loader:

        images = batch.to(
            DEVICE,
            dtype=torch.float32
        )

        optimizer.zero_grad()

        reconstruction, _ = model(
            images
        )

        loss = criterion(
            reconstruction,
            images
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(
            0
        )

        running_loss += (
            loss.item() *
            batch_size
        )

        n_samples += batch_size

    return (
        running_loss /
        max(n_samples, 1)
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_reconstruction_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion
) -> float:

    model.eval()

    running_loss = 0.0

    n_samples = 0

    for batch in loader:

        images = batch.to(
            DEVICE,
            dtype=torch.float32
        )

        reconstruction, _ = model(
            images
        )

        loss = criterion(
            reconstruction,
            images
        )

        batch_size = images.size(
            0
        )

        running_loss += (
            loss.item() *
            batch_size
        )

        n_samples += batch_size

    return (
        running_loss /
        max(n_samples, 1)
    )


# ============================================================
# FULL TRAINING LOOP
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader
) -> Tuple[nn.Module, pd.DataFrame]:

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    history = []

    best_val_loss = float(
        "inf"
    )

    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
        )

        val_loss = evaluate_reconstruction_loss(
            model,
            val_loader,
            criterion
        )

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss
        })

        print(
            f"  Epoch "
            f"{epoch:03d}/{NUM_EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        if (
            val_loss <
            best_val_loss - MIN_DELTA
        ):

            best_val_loss = val_loss

            best_state = {
                k: v.cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"  Early stopping at epoch "
                f"{epoch}."
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model.to(
        DEVICE
    )

    history_df = pd.DataFrame(
        history
    )

    return (
        model,
        history_df
    )


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def reconstruct_dataset(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = BATCH_SIZE
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray
]:
    """
    Ricostruisce un intero dataset.

    Returns:

        reconstructions
        image anomaly scores
        pixel anomaly maps
    """

    model.eval()

    tensor = torch.from_numpy(
        X
    )

    loader = DataLoader(
        tensor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    reconstructions = []

    scores = []

    anomaly_maps = []

    for images in loader:

        images = images.to(
            DEVICE
        )

        reconstruction, _ = model(
            images
        )

        error = torch.abs(
            images -
            reconstruction
        )

        mse = torch.mean(
            (
                images -
                reconstruction
            ) ** 2,
            dim=(1, 2, 3)
        )

        reconstructions.append(
            reconstruction.cpu().numpy()
        )

        scores.append(
            mse.cpu().numpy()
        )

        anomaly_maps.append(
            error.cpu().numpy()
        )

    reconstructions = np.concatenate(
        reconstructions,
        axis=0
    )

    scores = np.concatenate(
        scores,
        axis=0
    )

    anomaly_maps = np.concatenate(
        anomaly_maps,
        axis=0
    )

    return (
        reconstructions,
        scores,
        anomaly_maps
    )


# ============================================================
# IMAGE-LEVEL METRICS
# ============================================================

def evaluate_image_level(
    y_test: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Dict:

    y_pred = (
        scores >= threshold
    ).astype(
        np.int64
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp /
        max(tp + fn, 1)
    )

    specificity = (
        tn /
        max(tn + fp, 1)
    )

    precision = (
        tp /
        max(tp + fp, 1)
    )

    f1 = (
        2 *
        precision *
        sensitivity /
        max(
            precision +
            sensitivity,
            1e-8
        )
    )

    bacc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    auroc = roc_auc_score(
        y_test,
        scores
    )

    ap = average_precision_score(
        y_test,
        scores
    )

    fpr, tpr, roc_thresholds = roc_curve(
        y_test,
        scores
    )

    pr_precision, pr_recall, pr_thresholds = (
        precision_recall_curve(
            y_test,
            scores
        )
    )

    return {

        "auroc":
            auroc,

        "ap":
            ap,

        "threshold":
            threshold,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision_value":
            precision,

        "f1":
            f1,

        "bacc":
            bacc,

        "scores":
            scores,

        "y_pred":
            y_pred,

        "cm":
            cm,

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp),

        "fpr":
            fpr,

        "tpr":
            tpr,

        "roc_thresholds":
            roc_thresholds,

        "precision":
            pr_precision,

        "recall":
            pr_recall,

        "pr_thresholds":
            pr_thresholds
    }


# ============================================================
# THRESHOLD FROM HEALTHY VALIDATION
# ============================================================

def compute_image_threshold(
    healthy_scores: np.ndarray
) -> float:
    """
    Soglia image-level ottenuta esclusivamente
    dalle immagini healthy di validation.

    Il test non viene utilizzato.
    """

    threshold = np.percentile(
        healthy_scores,
        95
    )

    return float(
        threshold
    )


# ============================================================
# ANNOTATION FILE MAPPING
# ============================================================

def normalize_annotation_name(
    filename: str
) -> str:
    """
    Converte:

        BraTS2021_01467_seg_13.png

    in:

        BraTS2021_01467_flair_13.png
    """

    return re.sub(
        r"_seg_(\d+)\.png$",
        r"_flair_\1.png",
        filename
    )


def get_annotation_mapping() -> Dict[str, str]:
    """
    Costruisce il mapping:

        tumor image filename
            ->
        annotation filename

    e verifica che ogni tumor slice abbia
    una annotation corrispondente.
    """

    tumor_files = sorted(
        f
        for f in os.listdir(TUMOR_DIR)
        if f.endswith(".png")
    )

    annotation_files = sorted(
        f
        for f in os.listdir(ANNOTATION_DIR)
        if f.endswith(".png")
    )

    print(
        f"\n  Tumor images: "
        f"{len(tumor_files)}"
    )

    print(
        f"  Annotation masks: "
        f"{len(annotation_files)}"
    )

    if len(tumor_files) != len(
        annotation_files
    ):

        raise ValueError(
            "Numero di tumor images e "
            "annotation masks differente."
        )

    annotation_lookup = {
        f: f
        for f in annotation_files
    }

    mapping = {}

    missing = []

    for tumor_file in tumor_files:

        expected_annotation = (
            tumor_file
            .replace(
                "_flair_",
                "_seg_"
            )
        )

        if (
            expected_annotation
            not in annotation_lookup
        ):

            missing.append(
                expected_annotation
            )

        else:

            mapping[
                tumor_file
            ] = os.path.join(
                ANNOTATION_DIR,
                expected_annotation
            )

    if missing:

        raise ValueError(
            "Annotation mancanti. "
            f"Prime mancanti: {missing[:10]}"
        )

    print(
        "  ✓ Tumor ↔ annotation mapping verified."
    )

    return mapping


# ============================================================
# LOAD PIXEL GROUND TRUTH
# ============================================================

def load_mask(
    path: str
) -> np.ndarray:
    """
    Carica una segmentation mask.

    Qualsiasi valore > 0 viene considerato
    tumor pixel.

    Output:
        binary mask [64,64]
    """

    mask = np.array(
        Image.open(path)
    )

    if mask.ndim == 3:

        mask = mask[..., 0]

    mask = (
        mask > 0
    ).astype(
        np.uint8
    )

    return mask


def load_test_masks(
    tumor_files: List[str]
) -> np.ndarray:
    """
    Carica le mask nello stesso ordine
    delle tumor images.
    """

    mapping = get_annotation_mapping()

    masks = []

    for filename in tumor_files:

        mask_path = mapping[
            filename
        ]

        mask = load_mask(
            mask_path
        )

        masks.append(
            mask
        )

    masks = np.stack(
        masks,
        axis=0
    )

    return masks


# ============================================================
# PIXEL-LEVEL THRESHOLD
# ============================================================

def compute_pixel_threshold(
    healthy_anomaly_maps: np.ndarray
) -> float:
    """
    Threshold pixel-level ottenuto esclusivamente
    dalle anomaly maps delle immagini healthy
    di validation.

    Il 99th percentile dell'errore healthy
    viene utilizzato come soglia operativa.
    """

    threshold = np.percentile(
        healthy_anomaly_maps,
        99
    )

    return float(
        threshold
    )


# ============================================================
# PIXEL-LEVEL METRICS
# ============================================================

def evaluate_pixel_level(
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    threshold: float
) -> Dict:

    scores = anomaly_maps.reshape(
        -1
    )

    labels = masks.reshape(
        -1
    )

    pixel_auroc = roc_auc_score(
        labels,
        scores
    )

    pixel_ap = average_precision_score(
        labels,
        scores
    )

    binary_prediction = (
        anomaly_maps >= threshold
    ).astype(
        np.uint8
    )

    tp = np.sum(
        (
            binary_prediction == 1
        ) &
        (
            masks == 1
        )
    )

    fp = np.sum(
        (
            binary_prediction == 1
        ) &
        (
            masks == 0
        )
    )

    fn = np.sum(
        (
            binary_prediction == 0
        ) &
        (
            masks == 1
        )
    )

    tn = np.sum(
        (
            binary_prediction == 0
        ) &
        (
            masks == 0
        )
    )

    dice = (
        2 * tp /
        max(
            2 * tp +
            fp +
            fn,
            1
        )
    )

    iou = (
        tp /
        max(
            tp +
            fp +
            fn,
            1
        )
    )

    pixel_sensitivity = (
        tp /
        max(
            tp + fn,
            1
        )
    )

    pixel_specificity = (
        tn /
        max(
            tn + fp,
            1
        )
    )

    return {

        "pixel_auroc":
            pixel_auroc,

        "pixel_ap":
            pixel_ap,

        "pixel_threshold":
            threshold,

        "dice":
            float(dice),

        "iou":
            float(iou),

        "sensitivity":
            float(pixel_sensitivity),

        "specificity":
            float(pixel_specificity),

        "tp":
            int(tp),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tn":
            int(tn)
    }


# ============================================================
# TRAINING CURVE
# ============================================================

def plot_training_history(
    history: pd.DataFrame,
    out_dir: str
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        history["epoch"],
        history["train_loss"],
        label="Train"
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        label="Validation"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "MSE Reconstruction Loss"
    )

    ax.set_title(
        "CNN Autoencoder Training",
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "training_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# IMAGE-LEVEL RESULTS FIGURE
# ============================================================

def plot_image_level_results(
    metrics: Dict,
    y_test: np.ndarray,
    out_dir: str
) -> None:

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    ax = axes[0, 0]

    healthy = metrics["scores"][
        y_test == 0
    ]

    tumor = metrics["scores"][
        y_test == 1
    ]

    if len(healthy) > 1:

        sns.kdeplot(
            healthy,
            fill=True,
            alpha=0.4,
            label="Healthy",
            ax=ax
        )

    if len(tumor) > 1:

        sns.kdeplot(
            tumor,
            fill=True,
            alpha=0.4,
            label="Tumor",
            ax=ax
        )

    ax.axvline(
        metrics["threshold"],
        linestyle="--",
        color="black",
        linewidth=2,
        label=(
            "Validation threshold = "
            f"{metrics['threshold']:.5f}"
        )
    )

    ax.set_xlabel(
        "Reconstruction MSE"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "(A) Image-level Anomaly Scores",
        fontweight="bold"
    )

    ax.legend()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    ax = axes[0, 1]

    sns.heatmap(
        metrics["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=[
            "Healthy",
            "Tumor"
        ],
        yticklabels=[
            "Healthy",
            "Tumor"
        ]
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        "(B) Confusion Matrix",
        fontweight="bold"
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    ax = axes[1, 0]

    ax.plot(
        metrics["fpr"],
        metrics["tpr"],
        linewidth=2.5,
        label=(
            f"AUROC = "
            f"{metrics['auroc']:.3f}"
        )
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        alpha=0.6
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "(C) ROC Curve",
        fontweight="bold"
    )

    ax.legend()

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    ax = axes[1, 1]

    baseline = np.mean(
        y_test
    )

    ax.plot(
        metrics["recall"],
        metrics["precision"],
        linewidth=2.5,
        label=(
            f"AP = "
            f"{metrics['ap']:.3f}"
        )
    )

    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        alpha=0.6,
        label=(
            f"Baseline = "
            f"{baseline:.3f}"
        )
    )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "(D) Precision-Recall Curve",
        fontweight="bold"
    )

    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "image_level_results.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# PIXEL-LEVEL RESULTS
# ============================================================

def plot_pixel_level_results(
    pixel_metrics: Dict,
    out_dir: str
) -> None:

    data = [

        [
            "Pixel AUROC",
            f"{pixel_metrics['pixel_auroc']:.4f}"
        ],

        [
            "Pixel Average Precision",
            f"{pixel_metrics['pixel_ap']:.4f}"
        ],

        [
            "Dice",
            f"{pixel_metrics['dice']:.4f}"
        ],

        [
            "IoU",
            f"{pixel_metrics['iou']:.4f}"
        ],

        [
            "Pixel Sensitivity",
            f"{pixel_metrics['sensitivity']:.4f}"
        ],

        [
            "Pixel Specificity",
            f"{pixel_metrics['specificity']:.4f}"
        ],

        [
            "Pixel Threshold",
            f"{pixel_metrics['pixel_threshold']:.6f}"
        ]
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.axis(
        "off"
    )

    table = ax.table(
        cellText=data,
        colLabels=[
            "Metric",
            "Value"
        ],
        cellLoc="center",
        loc="center",
        colWidths=[
            0.45,
            0.25
        ]
    )

    table.auto_set_font_size(
        False
    )

    table.set_fontsize(
        11
    )

    table.scale(
        1,
        2.2
    )

    plt.title(
        "CNN Autoencoder — Pixel-Level Performance",
        fontweight="bold",
        pad=20
    )

    path = os.path.join(
        out_dir,
        "pixel_level_results.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# RECONSTRUCTION / ANOMALY MAP VISUALIZATION
# ============================================================

def plot_reconstruction_examples(
    X_test: np.ndarray,
    reconstructions: np.ndarray,
    anomaly_maps: np.ndarray,
    y_test: np.ndarray,
    test_tumor_indices: np.ndarray,
    test_masks: np.ndarray,
    out_dir: str,
    n_examples: int = 5
) -> None:
    """
    Salva esempi tumorali con:

        original
        reconstruction
        anomaly map
        ground truth
        overlay
    """

    if len(test_tumor_indices) == 0:

        return

    n = min(
        n_examples,
        len(test_tumor_indices)
    )

    selected = (
        test_tumor_indices[
            :n
        ]
    )

    fig, axes = plt.subplots(
        n,
        5,
        figsize=(15, 3 * n)
    )

    if n == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, idx in enumerate(
        selected
    ):

        original = X_test[
            idx,
            0
        ]

        reconstruction = (
            reconstructions[
                idx,
                0
            ]
        )

        anomaly_map = (
            anomaly_maps[
                idx,
                0
            ]
        )

        # Tumor masks are stored in the same
        # order as the tumor test subset.
        tumor_position = np.where(
            test_tumor_indices == idx
        )[0][0]

        mask = test_masks[
            tumor_position
        ]

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[row, 0].imshow(
            original,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 0].set_title(
            "Original"
        )

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        axes[row, 1].imshow(
            reconstruction,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 1].set_title(
            "Reconstruction"
        )

        # ----------------------------------------------------
        # Anomaly map
        # ----------------------------------------------------

        axes[row, 2].imshow(
            anomaly_map,
            cmap="hot"
        )

        axes[row, 2].set_title(
            "Anomaly Map"
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        axes[row, 3].imshow(
            mask,
            cmap="gray"
        )

        axes[row, 3].set_title(
            "Ground Truth"
        )

        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        axes[row, 4].imshow(
            original,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 4].imshow(
            anomaly_map,
            cmap="hot",
            alpha=0.45
        )

        axes[row, 4].contour(
            mask,
            levels=[0.5],
            colors="cyan",
            linewidths=1
        )

        axes[row, 4].set_title(
            "Anomaly + GT"
        )

        for col in range(5):

            axes[row, col].axis(
                "off"
            )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "reconstruction_anomaly_examples.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# ERROR ANALYSIS
# ============================================================

def save_error_analysis(
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    out_dir: str
) -> None:

    categories = np.full(
        len(y_test),
        "UNKNOWN",
        dtype=object
    )

    categories[
        (y_test == 0) &
        (y_pred == 0)
    ] = "TN"

    categories[
        (y_test == 0) &
        (y_pred == 1)
    ] = "FP"

    categories[
        (y_test == 1) &
        (y_pred == 0)
    ] = "FN"

    categories[
        (y_test == 1) &
        (y_pred == 1)
    ] = "TP"

    df = pd.DataFrame({

        "index":
            np.arange(
                len(y_test)
            ),

        "patient_id":
            patient_ids,

        "true_label":
            y_test,

        "predicted_label":
            y_pred,

        "category":
            categories,

        "anomaly_score":
            scores
    })

    path = os.path.join(
        out_dir,
        "error_analysis.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# SAVE IMAGE-LEVEL RESULTS
# ============================================================

def save_image_results(
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    patient_ids: np.ndarray,
    out_dir: str
) -> None:

    df = pd.DataFrame({

        "index":
            np.arange(
                len(y_test)
            ),

        "patient_id":
            patient_ids,

        "true_label":
            y_test,

        "predicted_label":
            y_pred,

        "anomaly_score":
            scores
    })

    path = os.path.join(
        out_dir,
        "image_level_results.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    image_metrics: Dict,
    pixel_metrics: Dict,
    history: pd.DataFrame,
    out_dir: str
) -> None:

    rows = {

        "Image_AUROC":
            image_metrics["auroc"],

        "Image_Average_Precision":
            image_metrics["ap"],

        "Image_F1":
            image_metrics["f1"],

        "Image_Sensitivity":
            image_metrics["sensitivity"],

        "Image_Specificity":
            image_metrics["specificity"],

        "Image_Balanced_Accuracy":
            image_metrics["bacc"],

        "Image_Threshold":
            image_metrics["threshold"],

        "Image_TN":
            image_metrics["tn"],

        "Image_FP":
            image_metrics["fp"],

        "Image_FN":
            image_metrics["fn"],

        "Image_TP":
            image_metrics["tp"],

        "Pixel_AUROC":
            pixel_metrics["pixel_auroc"],

        "Pixel_Average_Precision":
            pixel_metrics["pixel_ap"],

        "Pixel_Dice":
            pixel_metrics["dice"],

        "Pixel_IoU":
            pixel_metrics["iou"],

        "Pixel_Sensitivity":
            pixel_metrics["sensitivity"],

        "Pixel_Specificity":
            pixel_metrics["specificity"],

        "Pixel_Threshold":
            pixel_metrics["pixel_threshold"],

        "Epochs_Trained":
            len(history)
    }

    df = pd.DataFrame(
        [rows]
    )

    path = os.path.join(
        out_dir,
        "cnn_autoencoder_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# REPORT
# ============================================================

def save_report(
    model: nn.Module,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    image_metrics: Dict,
    pixel_metrics: Dict,
    history: pd.DataFrame,
    train_time: float,
    out_dir: str
) -> None:

    path = os.path.join(
        out_dir,
        "cnn_autoencoder_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "CNN AUTOENCODER — UNSUPERVISED DEEP BASELINE\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        f.write(
            "METHODOLOGY\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Training performed exclusively on healthy images.\n"
        )

        f.write(
            "Tumor labels were not used during training.\n"
        )

        f.write(
            "Validation contains healthy images only.\n"
        )

        f.write(
            "The test set was not used for model selection.\n"
        )

        f.write(
            "Tumor segmentation masks were used only for final\n"
        )

        f.write(
            "pixel-level evaluation.\n\n"
        )

        # ----------------------------------------------------
        # Architecture
        # ----------------------------------------------------

        f.write(
            "MODEL\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Architecture: Convolutional Autoencoder\n"
        )

        f.write(
            f"Latent dimension: {LATENT_DIM}\n"
        )

        f.write(
            "Input resolution: 64x64\n"
        )

        f.write(
            "Input channels: 1\n"
        )

        f.write(
            "Reconstruction activation: Sigmoid\n"
        )

        f.write(
            "Loss: Mean Squared Error\n"
        )

        f.write(
            "Optimizer: Adam\n"
        )

        f.write(
            f"Learning rate: {LEARNING_RATE}\n"
        )

        f.write(
            f"Weight decay: {WEIGHT_DECAY}\n"
        )

        f.write(
            f"Batch size: {BATCH_SIZE}\n"
        )

        f.write(
            f"Maximum epochs: {NUM_EPOCHS}\n"
        )

        f.write(
            f"Early stopping patience: {PATIENCE}\n\n"
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        f.write(
            "DATASET\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Train samples: {len(y_train)}\n"
        )

        f.write(
            f"Train healthy: {np.sum(y_train == 0)}\n"
        )

        f.write(
            f"Test samples: {len(y_test)}\n"
        )

        f.write(
            f"Test healthy: {np.sum(y_test == 0)}\n"
        )

        f.write(
            f"Test tumor: {np.sum(y_test == 1)}\n\n"
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        f.write(
            "TRAINING\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Training time: {train_time:.2f}s\n"
        )

        f.write(
            f"Epochs completed: {len(history)}\n"
        )

        f.write(
            f"Best validation loss: "
            f"{history['val_loss'].min():.8f}\n\n"
        )

        # ----------------------------------------------------
        # Image-level
        # ----------------------------------------------------

        f.write(
            "IMAGE-LEVEL RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"AUROC:                 "
            f"{image_metrics['auroc']:.4f}\n"
        )

        f.write(
            f"Average Precision:     "
            f"{image_metrics['ap']:.4f}\n"
        )

        f.write(
            f"F1:                    "
            f"{image_metrics['f1']:.4f}\n"
        )

        f.write(
            f"Sensitivity:           "
            f"{image_metrics['sensitivity']:.4f}\n"
        )

        f.write(
            f"Specificity:           "
            f"{image_metrics['specificity']:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy:     "
            f"{image_metrics['bacc']:.4f}\n"
        )

        f.write(
            f"Image threshold:       "
            f"{image_metrics['threshold']:.6f}\n\n"
        )

        # ----------------------------------------------------
        # Pixel-level
        # ----------------------------------------------------

        f.write(
            "PIXEL-LEVEL RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Pixel AUROC:           "
            f"{pixel_metrics['pixel_auroc']:.4f}\n"
        )

        f.write(
            f"Pixel Average Precision: "
            f"{pixel_metrics['pixel_ap']:.4f}\n"
        )

        f.write(
            f"Dice:                  "
            f"{pixel_metrics['dice']:.4f}\n"
        )

        f.write(
            f"IoU:                   "
            f"{pixel_metrics['iou']:.4f}\n"
        )

        f.write(
            f"Pixel Sensitivity:     "
            f"{pixel_metrics['sensitivity']:.4f}\n"
        )

        f.write(
            f"Pixel Specificity:     "
            f"{pixel_metrics['specificity']:.4f}\n"
        )

        f.write(
            f"Pixel threshold:       "
            f"{pixel_metrics['pixel_threshold']:.6f}\n\n"
        )

        # ----------------------------------------------------
        # Leakage statement
        # ----------------------------------------------------

        f.write(
            "LEAKAGE CONTROL\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Tumor labels were not used during optimization.\n"
        )

        f.write(
            "Tumor segmentation masks were not used during optimization.\n"
        )

        f.write(
            "Image-level threshold was computed from healthy validation data.\n"
        )

        f.write(
            "Pixel-level threshold was computed from healthy validation data.\n"
        )

        f.write(
            "The test set was reserved for final evaluation.\n\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "End of report\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

    print(
        f"  ✓ Saved: {path}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment() -> None:

    total_start = time.time()

    print(
        "\n" +
        "=" * 70
    )

    print(
        " CNN AUTOENCODER — UNSUPERVISED DEEP BASELINE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print(
        "\n[1/8] Loading datasets..."
    )

    t0 = time.time()

    train_ds = get_dataset(
        "brats",
        mode="train"
    )

    test_ds = get_dataset(
        "brats",
        mode="test"
    )

    print(
        f"  ✓ Train: {len(train_ds)}"
    )

    print(
        f"  ✓ Test:  {len(test_ds)}"
    )

    print("\n========== DATALOADER STRUCTURE CHECK ==========")

    print("TRAIN keys:")
    print(train_ds[0].keys())

    print("\nTEST keys:")
    print(test_ds[0].keys())

    print("\nTEST sample:")
    print(test_ds[0])

    print("===============================================\n")

    # ========================================================
    # 2. EXTRACT DATA
    # ========================================================

    print(
        "\n[2/8] Extracting images..."
    )

    X_train, y_train = extract_features(
        train_ds
    )

    X_test, y_test = extract_features(
        test_ds
    )

    patient_ids_test = extract_patient_ids(
        test_ds
    )

    '''test_masks = np.stack([
        np.asarray(test_ds[i]["mask"])
        for i in range(len(test_ds))
    ])

    # Da (N, 1, 64, 64) a (N, 64, 64)
    test_masks = test_masks[:, 0, :, :]'''



    test_masks = []

    for i in range(len(test_ds)):

        mask = test_ds[i]["mask"]

        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()

        mask = np.asarray(mask)

        # Remove channel dimension if present
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]

        # Resize mask to match the image resolution
        if mask.shape != X_test.shape[2:]:
            mask = np.array(
                Image.fromarray(
                    mask.astype(np.uint8)
                ).resize(
                (
                    X_test.shape[3],
                    X_test.shape[2]
                ),
                resample=Image.Resampling.NEAREST
            )
        )

        # Convert to binary tumor mask
        mask = (
            mask > 0
        ).astype(
            np.uint8
        )

        test_masks.append(mask)

    test_masks = np.stack(
        test_masks,
        axis=0
    )

    print(
        f"  ✓ Test masks: {test_masks.shape}"
    )

    print(
        f"  ✓ X_train: {X_train.shape}"
    )

    print(
        f"  ✓ X_test:  {X_test.shape}"
    )

    print(
        f"  ✓ Train labels: "
        f"{np.unique(y_train, return_counts=True)}"
    )

    print(
        f"  ✓ Test labels: "
        f"{np.unique(y_test, return_counts=True)}"
    )

    if not np.all(
        y_train == 0
    ):

        raise ValueError(
            "TRAIN contiene campioni tumorali. "
            "Il CNN Autoencoder deve essere "
            "addestrato esclusivamente su healthy."
        )

    print(
        "  ✓ TRAIN verified: healthy only."
    )

    # ========================================================
    # CHECK IMAGE RANGE
    # ========================================================

    if (
        X_train.min() < 0 or
        X_train.max() > 1
    ):

        raise ValueError(
            "Le immagini devono essere "
            "normalizzate in [0,1]."
        )

    print(
        f"  ✓ Image range: "
        f"[{X_train.min():.3f}, "
        f"{X_train.max():.3f}]"
    )

    # ========================================================
    # 3. TRAIN / VALIDATION SPLIT
    # ========================================================

    print(
        "\n[3/8] Creating healthy train/validation split..."
    )

    train_indices, val_indices = (
        create_train_validation_split(
            len(X_train)
        )
    )

    X_train_model = X_train[
        train_indices
    ]

    X_val = X_train[
        val_indices
    ]

    print(
        f"  ✓ Training healthy: "
        f"{len(X_train_model)}"
    )

    print(
        f"  ✓ Validation healthy: "
        f"{len(X_val)}"
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_tensor = torch.from_numpy(
        X_train_model
    )

    val_tensor = torch.from_numpy(
        X_val
    )

    train_loader = DataLoader(
        train_tensor,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_tensor,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ========================================================
    # 4. MODEL
    # ========================================================

    print(
        "\n[4/8] Building CNN Autoencoder..."
    )

    model = ConvAutoencoder(
        latent_dim=LATENT_DIM
    ).to(
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"  ✓ Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"  ✓ Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ========================================================
    # 5. TRAINING
    # ========================================================

    print(
        "\n[5/8] Training..."
    )

    t0 = time.time()

    model, history = train_model(
        model,
        train_loader,
        val_loader
    )

    train_time = (
        time.time() - t0
    )

    history_path = os.path.join(
        OUT_DIR,
        "training_history.csv"
    )

    history.to_csv(
        history_path,
        index=False
    )

    print(
        f"  ✓ Training time: "
        f"{train_time:.2f}s"
    )

    print(
        f"  ✓ Best validation loss: "
        f"{history['val_loss'].min():.8f}"
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(
        OUT_DIR,
        "cnn_autoencoder_best.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "latent_dim":
                LATENT_DIM,

            "seed":
                SEED,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY
        },
        model_path
    )

    print(
        f"  ✓ Saved: {model_path}"
    )

    # ========================================================
    # 6. INFERENCE
    # ========================================================

    print(
        "\n[6/8] Computing reconstruction errors..."
    )

    t0 = time.time()

    (
        train_recon,
        train_scores,
        train_maps
    ) = reconstruct_dataset(
        model,
        X_train_model
    )

    (
        val_recon,
        val_scores,
        val_maps
    ) = reconstruct_dataset(
        model,
        X_val
    )

    (
        test_recon,
        test_scores,
        test_maps
    ) = reconstruct_dataset(
        model,
        X_test
    )

    inference_time = (
        time.time() - t0
    )

    print(
        f"  ✓ Inference time: "
        f"{inference_time:.2f}s"
    )

    # ========================================================
    # 7. THRESHOLDS
    # ========================================================

    print(
        "\n[7/8] Computing validation-only thresholds..."
    )

    image_threshold = compute_image_threshold(
        val_scores
    )

    pixel_threshold = compute_pixel_threshold(
        val_maps
    )

    print(
        f"  ✓ Image threshold: "
        f"{image_threshold:.6f}"
    )

    print(
        f"  ✓ Pixel threshold: "
        f"{pixel_threshold:.6f}"
    )

    # ========================================================
    # IMAGE-LEVEL EVALUATION
    # ========================================================

    image_metrics = evaluate_image_level(
        y_test=y_test,
        scores=test_scores,
        threshold=image_threshold
    )

    
    # ========================================================
    # TUMOR SUBSET
    # ========================================================

    tumor_indices = np.where(
        y_test == 1
    )[0]

    tumor_anomaly_maps = test_maps[
        tumor_indices,
        0
    ]

    tumor_masks = test_masks[
        tumor_indices
    ]

    # ========================================================
    # FINAL SHAPE CHECK
    # ========================================================

    print(
        f"  ✓ Tumor anomaly maps: "
        f"{tumor_anomaly_maps.shape}"
    )

    print(
        f"  ✓ Tumor masks:        "
        f"{tumor_masks.shape}"
    )

    if tumor_anomaly_maps.shape != tumor_masks.shape:
        raise ValueError(
            "Anomaly maps e annotation non compatibili: "
            f"{tumor_anomaly_maps.shape} vs "
            f"{tumor_masks.shape}"
        )

    # ========================================================
    # PIXEL-LEVEL EVALUATION
    # ========================================================

    pixel_metrics = evaluate_pixel_level(
        anomaly_maps=tumor_anomaly_maps,
        masks=tumor_masks,
        threshold=pixel_threshold
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        " IMAGE-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"AUROC:              "
        f"{image_metrics['auroc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{image_metrics['ap']:.4f}"
    )

    print(
        f"F1:                 "
        f"{image_metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity:        "
        f"{image_metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity:        "
        f"{image_metrics['specificity']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{image_metrics['bacc']:.4f}"
    )

    print(
        f"Threshold:          "
        f"{image_metrics['threshold']:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"  TN: {image_metrics['tn']}"
    )

    print(
        f"  FP: {image_metrics['fp']}"
    )

    print(
        f"  FN: {image_metrics['fn']}"
    )

    print(
        f"  TP: {image_metrics['tp']}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        " PIXEL-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Pixel AUROC:        "
        f"{pixel_metrics['pixel_auroc']:.4f}"
    )

    print(
        f"Pixel AP:           "
        f"{pixel_metrics['pixel_ap']:.4f}"
    )

    print(
        f"Dice:               "
        f"{pixel_metrics['dice']:.4f}"
    )

    print(
        f"IoU:                "
        f"{pixel_metrics['iou']:.4f}"
    )

    print(
        f"Pixel Sensitivity:  "
        f"{pixel_metrics['sensitivity']:.4f}"
    )

    print(
        f"Pixel Specificity:  "
        f"{pixel_metrics['specificity']:.4f}"
    )

    print(
        f"Pixel threshold:    "
        f"{pixel_metrics['pixel_threshold']:.6f}"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    print(
        "\n[VISUALIZATION] Generating plots..."
    )

    plot_training_history(
        history,
        OUT_DIR
    )

    plot_image_level_results(
        image_metrics,
        y_test,
        OUT_DIR
    )

    plot_pixel_level_results(
        pixel_metrics,
        OUT_DIR
    )

    plot_reconstruction_examples(
        X_test=X_test,
        reconstructions=test_recon,
        anomaly_maps=test_maps,
        y_test=y_test,
        test_tumor_indices=tumor_indices,
        test_masks=test_masks,
        out_dir=OUT_DIR,
        n_examples=5
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print(
        "\n[RESULTS] Saving CSV files..."
    )

    save_image_results(
        y_test=y_test,
        scores=test_scores,
        y_pred=image_metrics["y_pred"],
        patient_ids=patient_ids_test,
        out_dir=OUT_DIR
    )

    save_error_analysis(
        y_test=y_test,
        scores=test_scores,
        y_pred=image_metrics["y_pred"],
        patient_ids=patient_ids_test,
        out_dir=OUT_DIR
    )

    save_metrics(
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        history=history,
        out_dir=OUT_DIR
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n[REPORTING] Saving report..."
    )

    save_report(
        model=model,
        X_train=X_train_model,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        image_metrics=image_metrics,
        pixel_metrics=pixel_metrics,
        history=history,
        train_time=train_time,
        out_dir=OUT_DIR
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.time() -
        total_start
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        " EXPERIMENT COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        f"Total time: "
        f"{total_time:.2f}s"
    )

    print(
        f"Output directory: "
        f"{OUT_DIR}/"
    )

    print(
        "\nGenerated files:"
    )

    print(
        "  • cnn_autoencoder_best.pt"
    )

    print(
        "  • training_history.csv"
    )

    print(
        "  • training_curve.png"
    )

    print(
        "  • image_level_results.png"
    )

    print(
        "  • pixel_level_results.png"
    )

    print(
        "  • reconstruction_anomaly_examples.png"
    )

    print(
        "  • image_level_results.csv"
    )

    print(
        "  • error_analysis.csv"
    )

    print(
        "  • cnn_autoencoder_metrics.csv"
    )

    print(
        "  • cnn_autoencoder_report.txt"
    )

    print(
        "\nMethodology:"
    )

    print(
        "  • Training: HEALTHY only"
    )

    print(
        "  • Validation: HEALTHY only"
    )

    print(
        "  • Test: HEALTHY + TUMOR"
    )

    print(
        "  • Tumor labels during training: NO"
    )

    print(
        "  • Segmentation masks during training: NO"
    )

    print(
        "  • Image-level score: reconstruction MSE"
    )

    print(
        "  • Pixel-level score: absolute reconstruction error"
    )

    print(
        "  • Image threshold: validation healthy P95"
    )

    print(
        "  • Pixel threshold: validation healthy P99"
    )

    print(
        "  • Test threshold tuning: NO"
    )

    print(
        "  • Patient-level evaluation: NO"
    )

    print(
        "\n" +
        "=" * 70 +
        "\n"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_experiment()