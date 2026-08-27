# ============================================================
# CUTPASTE — SELF-SUPERVISED ANOMALY DETECTION
# ============================================================
#
# Based on:
#
# Li et al.
# "CutPaste: Self-Supervised Learning for Anomaly Detection
# and Localization"
# CVPR 2021
#
# EXPERIMENTAL PROTOCOL
# ------------------------------------------------------------
#
# DATA:
#
#   BraTS2021/
#   ├── train/
#   │   └── healthy images
#   │
#   └── test/
#       ├── normal/
#       ├── tumor/
#       └── annotation/
#
#
# TRAINING:
#
#   original healthy image
#          |
#          +---- Normal        -> class 0
#          |
#          +---- CutPaste      -> class 1
#          |
#          +---- CutPaste-SCAR -> class 2
#
# The CutPaste classifier is trained exclusively using
# healthy training images.
#
#
# TRAIN / VALIDATION:
#
# The original healthy TRAIN set is split into:
#
#   85% -> self-supervised training
#   15% -> validation
#
# The validation set is healthy-only.
#
# Validation is used for:
#
#   1. early stopping
#   2. model selection
#   3. anomaly-score threshold selection
#
#
# TEST:
#
# Healthy + tumor images are used only once at the end.
#
#
# ANOMALY DETECTION:
#
#   trained backbone
#          ↓
#   deep feature representation
#          ↓
#   healthy train features
#          ↓
#   Gaussian density estimation
#          ↓
#   Mahalanobis distance
#          ↓
#   anomaly score
#
#
# IMPORTANT:
#
# No tumor information is used during:
#
#   - self-supervised training
#   - validation
#   - model selection
#   - threshold selection
#
# Tumor labels are used only for final evaluation.
#
# ============================================================


import os
import sys
import time
import random
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    balanced_accuracy_score,
)

from scipy.linalg import pinvh


# ============================================================
# PATH
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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

#NUM_WORKERS = 0
NUM_WORKERS = 2

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

VAL_RATIO = 0.15

PATIENCE = 10

MIN_DELTA = 1e-5

N_CLASSES = 3

GAUSSIAN_REGULARIZATION = 1e-5

THRESHOLD_PERCENTILE = 95

# ------------------------------------------------------------
# CUTPASTE PARAMETERS
# ------------------------------------------------------------

CUTPASTE_AREA_MIN = 0.02

CUTPASTE_AREA_MAX = 0.15

CUTPASTE_ASPECT_MIN = 0.3

CUTPASTE_ASPECT_MAX = 3.0

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUT_DIR = os.path.join(
    "results",
    "cutpaste"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(
    seed: int = SEED
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

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

plt.rcParams[
    "figure.facecolor"
] = "white"


# ============================================================
# CUTPASTE
# ============================================================

class CutPaste:

    """
    CutPaste transformation.

    A rectangular patch is extracted from the image
    and pasted at another random location.
    """

    def __init__(
        self,
        area_ratio_min: float = CUTPASTE_AREA_MIN,
        area_ratio_max: float = CUTPASTE_AREA_MAX,
        aspect_ratio_min: float = CUTPASTE_ASPECT_MIN,
        aspect_ratio_max: float = CUTPASTE_ASPECT_MAX,
    ):

        self.area_ratio_min = area_ratio_min

        self.area_ratio_max = area_ratio_max

        self.aspect_ratio_min = aspect_ratio_min

        self.aspect_ratio_max = aspect_ratio_max

    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:

        if image.ndim != 3:

            raise ValueError(
                "CutPaste richiede un'immagine "
                "[C,H,W]."
            )

        _, height, width = image.shape

        # ----------------------------------------------------
        # PATCH AREA
        # ----------------------------------------------------

        area = random.uniform(
            self.area_ratio_min,
            self.area_ratio_max
        ) * height * width

        # ----------------------------------------------------
        # PATCH ASPECT RATIO
        # ----------------------------------------------------

        aspect_ratio = random.uniform(
            self.aspect_ratio_min,
            self.aspect_ratio_max
        )

        patch_h = int(
            np.sqrt(
                area / aspect_ratio
            )
        )

        patch_w = int(
            np.sqrt(
                area * aspect_ratio
            )
        )

        patch_h = max(
            1,
            min(
                patch_h,
                height - 1
            )
        )

        patch_w = max(
            1,
            min(
                patch_w,
                width - 1
            )
        )

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        src_y = random.randint(
            0,
            height - patch_h
        )

        src_x = random.randint(
            0,
            width - patch_w
        )

        patch = image[
            :,
            src_y:src_y + patch_h,
            src_x:src_x + patch_w
        ].clone()

        # ----------------------------------------------------
        # DESTINATION
        # ----------------------------------------------------

        dst_y = random.randint(
            0,
            height - patch_h
        )

        dst_x = random.randint(
            0,
            width - patch_w
        )

        # ----------------------------------------------------
        # PASTE
        # ----------------------------------------------------

        result = image.clone()

        result[
            :,
            dst_y:dst_y + patch_h,
            dst_x:dst_x + patch_w
        ] = patch

        return result


# ============================================================
# CUTPASTE-SCAR
# ============================================================

class CutPasteScar:

    """
    CutPaste-SCAR transformation.

    Generates a thin elongated patch and pastes it
    at a different location.
    """

    def __init__(
        self,
        width_ratio: float = 0.125
    ):

        self.width_ratio = width_ratio

    def __call__(
        self,
        image: torch.Tensor
    ) -> torch.Tensor:

        if image.ndim != 3:

            raise ValueError(
                "CutPasteScar richiede un'immagine "
                "[C,H,W]."
            )

        _, height, width = image.shape

        horizontal = (
            random.random() < 0.5
        )

        # ----------------------------------------------------
        # HORIZONTAL SCAR
        # ----------------------------------------------------

        if horizontal:

            patch_h = random.randint(
                1,
                max(
                    1,
                    height // 8
                )
            )

            patch_w = random.randint(
                max(
                    2,
                    width // 8
                ),
                max(
                    2,
                    width // 2
                )
            )

        # ----------------------------------------------------
        # VERTICAL SCAR
        # ----------------------------------------------------

        else:

            patch_h = random.randint(
                max(
                    2,
                    height // 8
                ),
                max(
                    2,
                    height // 2
                )
            )

            patch_w = random.randint(
                1,
                max(
                    1,
                    width // 8
                )
            )

        patch_h = min(
            patch_h,
            height - 1
        )

        patch_w = min(
            patch_w,
            width - 1
        )

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        src_y = random.randint(
            0,
            height - patch_h
        )

        src_x = random.randint(
            0,
            width - patch_w
        )

        patch = image[
            :,
            src_y:src_y + patch_h,
            src_x:src_x + patch_w
        ].clone()

        # ----------------------------------------------------
        # DESTINATION
        # ----------------------------------------------------

        dst_y = random.randint(
            0,
            height - patch_h
        )

        dst_x = random.randint(
            0,
            width - patch_w
        )

        # ----------------------------------------------------
        # PASTE
        # ----------------------------------------------------

        result = image.clone()

        result[
            :,
            dst_y:dst_y + patch_h,
            dst_x:dst_x + patch_w
        ] = patch

        return result


# ============================================================
# SELF-SUPERVISED DATASET
# ============================================================

class CutPasteDataset(
    Dataset
):

    """
    Converts a healthy base dataset into the
    three-class CutPaste self-supervised dataset.

    For each healthy image:

        class 0 -> original
        class 1 -> CutPaste
        class 2 -> CutPaste-SCAR
    """

    def __init__(
        self,
        base_dataset
    ):

        super().__init__()

        self.base_dataset = base_dataset

        self.cutpaste = CutPaste()

        self.scar = CutPasteScar()

    def __len__(
        self
    ) -> int:

        return len(
            self.base_dataset
        )

    def __getitem__(
        self,
        index: int
    ) -> Dict:

        sample = self.base_dataset[
            index
        ]

        image = sample[
            "img"
        ].float()

        # ----------------------------------------------------
        # ORIGINAL
        # ----------------------------------------------------

        normal = image.clone()

        # ----------------------------------------------------
        # CUTPASTE
        # ----------------------------------------------------

        cutpaste = self.cutpaste(
            image
        )

        # ----------------------------------------------------
        # SCAR
        # ----------------------------------------------------

        scar = self.scar(
            image
        )

        images = torch.stack(
            [
                normal,
                cutpaste,
                scar
            ],
            dim=0
        )

        labels = torch.tensor(
            [
                0,
                1,
                2
            ],
            dtype=torch.long
        )

        return {
            "images": images,
            "labels": labels
        }


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    n_samples: int,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED
) -> Tuple[
    np.ndarray,
    np.ndarray
]:

    """
    Splits the healthy training set into:

        training healthy
        validation healthy

    No test information is used.
    """

    if not (
        0.0 < val_ratio < 1.0
    ):

        raise ValueError(
            "val_ratio deve essere compreso "
            "tra 0 e 1."
        )

    if n_samples < 2:

        raise ValueError(
            "Sono necessari almeno 2 campioni "
            "per creare train/validation."
        )

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

    n_val = max(
        1,
        n_val
    )

    n_val = min(
        n_samples - 1,
        n_val
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
# MODEL
# ============================================================

class CutPasteModel(
    nn.Module
):

    """
    ResNet-18 adapted to grayscale MRI.

    The final classification layer predicts:

        0 = Normal
        1 = CutPaste
        2 = CutPaste-SCAR
    """

    def __init__(
        self
    ):

        super().__init__()

        backbone = models.resnet18(
            weights=None
        )

        # ----------------------------------------------------
        # GRAYSCALE INPUT
        # ----------------------------------------------------

        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        feature_dim = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.feature_dim = (
            feature_dim
        )

        self.classifier = nn.Linear(
            feature_dim,
            N_CLASSES
        )

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor
    ]:

        features = self.backbone(
            x
        )

        logits = self.classifier(
            features
        )

        return (
            logits,
            features
        )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_cutpaste(
    model: nn.Module,
    loader: DataLoader
) -> Tuple[
    float,
    float
]:

    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0

    correct = 0

    total = 0

    for batch in loader:

        images = batch[
            "images"
        ].to(
            DEVICE
        )

        labels = batch[
            "labels"
        ].to(
            DEVICE
        )

        batch_size = (
            images.shape[0]
        )

        images = images.reshape(
            batch_size * 3,
            *images.shape[2:]
        )

        labels = labels.reshape(
            -1
        )

        logits, _ = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        running_loss += (
            loss.item()
            *
            labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += (
            labels.size(0)
        )

    loss = (
        running_loss /
        max(total, 1)
    )

    accuracy = (
        correct /
        max(total, 1)
    )

    return (
        loss,
        accuracy
    )


# ============================================================
# SELF-SUPERVISED TRAINING
# ============================================================

def train_cutpaste(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = NUM_EPOCHS
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float(
        "inf"
    )

    best_state = deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

    for epoch in range(
        epochs
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        for batch in train_loader:

            images = batch[
                "images"
            ].to(
                DEVICE
            )

            labels = batch[
                "labels"
            ].to(
                DEVICE
            )

            # ------------------------------------------------
            # [B, 3, C, H, W]
            # ------------------------------------------------

            batch_size = (
                images.shape[0]
            )

            images = images.reshape(
                batch_size * 3,
                *images.shape[2:]
            )

            labels = labels.reshape(
                -1
            )

            optimizer.zero_grad()

            logits, _ = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                *
                labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += (
                labels.size(0)
            )

        train_loss = (
            running_loss /
            max(total, 1)
        )

        train_accuracy = (
            correct /
            max(total, 1)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = (
            validate_cutpaste(
                model,
                val_loader
            )
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "val_loss"
        ].append(
            val_loss
        )

        history[
            "val_accuracy"
        ].append(
            val_accuracy
        )

        print(
            f"Epoch "
            f"{epoch + 1:03d}/"
            f"{epochs:03d} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy:.4f} | "
            f"Val Loss: "
            f"{val_loss:.4f} | "
            f"Val Acc: "
            f"{val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < (
            best_val_loss - MIN_DELTA
        ):

            best_val_loss = val_loss

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch + 1}."
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    return (
        history,
        best_val_loss
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset,
    batch_size: int = BATCH_SIZE
) -> Tuple[
    np.ndarray,
    np.ndarray
]:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model.eval()

    features = []

    labels = []

    for batch in loader:

        images = batch[
            "img"
        ].float().to(
            DEVICE
        )

        batch_labels = (
            batch[
                "label"
            ]
            .cpu()
            .numpy()
        )

        _, batch_features = model(
            images
        )

        features.append(
            batch_features.cpu().numpy()
        )

        labels.append(
            batch_labels
        )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return (
        features,
        labels
    )


# ============================================================
# GAUSSIAN DENSITY ESTIMATOR
# ============================================================

class GaussianDensityEstimator:

    """
    Gaussian density model in feature space.

    The covariance matrix is regularized and inverted
    using the Moore-Penrose pseudoinverse.
    """

    def __init__(
        self,
        regularization: float = GAUSSIAN_REGULARIZATION
    ):

        self.regularization = (
            regularization
        )

        self.mean = None

        self.covariance = None

        self.precision = None

    def fit(
        self,
        features: np.ndarray
    ):

        if features.ndim != 2:

            raise ValueError(
                "features deve avere forma [N,D]."
            )

        if features.shape[0] < 2:

            raise ValueError(
                "Servono almeno due campioni "
                "per stimare la covarianza."
            )

        self.mean = np.mean(
            features,
            axis=0
        )

        covariance = np.cov(
            features,
            rowvar=False
        )

        # ----------------------------------------------------
        # HANDLE 1D CASE
        # ----------------------------------------------------

        if covariance.ndim == 0:

            covariance = np.array(
                [[float(covariance)]]
            )

        covariance = (
            covariance
            +
            self.regularization
            *
            np.eye(
                covariance.shape[0]
            )
        )

        self.covariance = covariance

        self.precision = pinvh(
            covariance
        )

        return self

    def score(
        self,
        features: np.ndarray
    ) -> np.ndarray:

        if (
            self.mean is None
            or
            self.precision is None
        ):

            raise RuntimeError(
                "GaussianDensityEstimator "
                "non è stato fittato."
            )

        centered = (
            features
            -
            self.mean
        )

        scores = np.einsum(
            "ij,jk,ik->i",
            centered,
            self.precision,
            centered
        )

        return scores


# ============================================================
# THRESHOLD
# ============================================================

def compute_threshold(
    healthy_scores: np.ndarray,
    percentile: float = THRESHOLD_PERCENTILE
) -> float:

    if healthy_scores.size == 0:

        raise ValueError(
            "healthy_scores è vuoto."
        )

    return float(
        np.percentile(
            healthy_scores,
            percentile
        )
    )


# ============================================================
# IMAGE-LEVEL EVALUATION
# ============================================================

def evaluate_image_level(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float
) -> Dict:

    y_true = np.asarray(
        y_true
    )

    scores = np.asarray(
        scores
    )

    if len(y_true) != len(scores):

        raise ValueError(
            "y_true e scores devono avere "
            "la stessa lunghezza."
        )

    y_pred = (
        scores >= threshold
    ).astype(
        int
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    # --------------------------------------------------------
    # DISCRIMINATION METRICS
    # --------------------------------------------------------

    auroc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    # --------------------------------------------------------
    # THRESHOLD METRICS
    # --------------------------------------------------------

    sensitivity = (
        tp /
        (
            tp +
            fn +
            1e-8
        )
    )

    specificity = (
        tn /
        (
            tn +
            fp +
            1e-8
        )
    )

    precision = (
        tp /
        (
            tp +
            fp +
            1e-8
        )
    )

    f1 = (
        2.0
        *
        precision
        *
        sensitivity
        /
        (
            precision
            +
            sensitivity
            +
            1e-8
        )
    )

    bacc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = (
        roc_curve(
            y_true,
            scores
        )
    )

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    return {

        "auroc": float(auroc),

        "ap": float(ap),

        "f1": float(f1),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "precision": float(
            precision
        ),

        "bacc": float(
            bacc
        ),

        "threshold": float(
            threshold
        ),

        "scores": scores,

        "y_pred": y_pred,

        "cm": cm,

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),

        "fpr": fpr,

        "tpr": tpr,

        "roc_thresholds": roc_thresholds,

        "precision_curve": precision_curve,

        "recall": recall_curve,

        "pr_thresholds": pr_thresholds
    }


# ============================================================
# TRAINING CURVES
# ============================================================

def plot_training_history(
    history: Dict,
    out_dir: str
) -> None:

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    epochs = np.arange(
        1,
        len(
            history["train_loss"]
        ) + 1
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_loss"],
        linewidth=2,
        label="Train"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        linewidth=2,
        label="Validation"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.title(
        "CutPaste Self-Supervised Loss"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "training_loss.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["train_accuracy"],
        linewidth=2,
        label="Train"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        linewidth=2,
        label="Validation"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "CutPaste Self-Supervised Accuracy"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "training_accuracy.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )


# ============================================================
# RESULT PLOTS
# ============================================================

def plot_results(
    metrics: Dict,
    y_test: np.ndarray,
    out_dir: str
) -> None:

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    healthy_scores = metrics[
        "scores"
    ][
        y_test == 0
    ]

    tumor_scores = metrics[
        "scores"
    ][
        y_test == 1
    ]

    if len(healthy_scores) > 1:

        sns.kdeplot(
            healthy_scores,
            fill=True,
            alpha=0.4,
            label="Healthy"
        )

    if len(tumor_scores) > 1:

        sns.kdeplot(
            tumor_scores,
            fill=True,
            alpha=0.4,
            label="Tumor"
        )

    plt.axvline(
        metrics["threshold"],
        linestyle="--",
        color="black",
        label="Validation P95"
    )

    plt.xlabel(
        "Mahalanobis Anomaly Score"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Anomaly Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "score_distribution.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        metrics["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Healthy",
            "Tumor"
        ],
        yticklabels=[
            "Healthy",
            "Tumor"
        ]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        metrics["fpr"],
        metrics["tpr"],
        linewidth=2,
        label=(
            f"AUROC = "
            f"{metrics['auroc']:.3f}"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )

    # --------------------------------------------------------
    # PR
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    baseline = np.mean(
        y_test
    )

    plt.plot(
        metrics["recall"],
        metrics["precision_curve"],
        linewidth=2,
        label=(
            f"AP = "
            f"{metrics['ap']:.3f}"
        )
    )

    plt.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label=(
            f"Baseline = "
            f"{baseline:.3f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    path = os.path.join(
        out_dir,
        "precision_recall_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"✓ Saved: {path}"
    )


# ============================================================
# SAVE METRICS
# ============================================================

def save_metrics(
    metrics: Dict,
    out_dir: str
) -> None:

    rows = {

        "AUROC":
            metrics["auroc"],

        "Average_Precision":
            metrics["ap"],

        "F1":
            metrics["f1"],

        "Sensitivity":
            metrics["sensitivity"],

        "Specificity":
            metrics["specificity"],

        "Precision":
            metrics["precision"],

        "Balanced_Accuracy":
            metrics["bacc"],

        "Threshold":
            metrics["threshold"],

        "TN":
            metrics["tn"],

        "FP":
            metrics["fp"],

        "FN":
            metrics["fn"],

        "TP":
            metrics["tp"]
    }

    df = pd.DataFrame(
        [rows]
    )

    path = os.path.join(
        out_dir,
        "cutpaste_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"✓ Saved: {path}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    metrics: Dict,
    history: Dict,
    train_features: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    train_time: float
) -> None:

    path = os.path.join(
        out_dir,
        "cutpaste_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 +
            "\n"
        )

        f.write(
            "CUTPASTE — SELF-SUPERVISED "
            "ANOMALY DETECTION\n"
        )

        f.write(
            "=" * 70 +
            "\n\n"
        )

        # ----------------------------------------------------
        # METHODOLOGY
        # ----------------------------------------------------

        f.write(
            "METHODOLOGY\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            "Training images: healthy only.\n"
        )

        f.write(
            "Validation images: healthy only.\n"
        )

        f.write(
            "Test images: healthy + tumor.\n"
        )

        f.write(
            "Tumor labels were not used during "
            "training or validation.\n\n"
        )

        f.write(
            "Self-supervised classes:\n"
        )

        f.write(
            "  0 = Normal\n"
        )

        f.write(
            "  1 = CutPaste\n"
        )

        f.write(
            "  2 = CutPaste-SCAR\n\n"
        )

        f.write(
            "Feature representation: ResNet-18 "
            "penultimate layer.\n"
        )

        f.write(
            "Density estimator: multivariate Gaussian.\n"
        )

        f.write(
            "Anomaly score: squared Mahalanobis distance.\n"
        )

        f.write(
            "Threshold: validation healthy P95.\n\n"
        )

        # ----------------------------------------------------
        # DATA
        # ----------------------------------------------------

        f.write(
            "DATA\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}\n"
        )

        f.write(
            "Channels: 1\n"
        )

        f.write(
            "Image range: [0,1]\n"
        )

        f.write(
            f"Train healthy: {len(train_indices)}\n"
        )

        f.write(
            f"Validation healthy: {len(val_indices)}\n"
        )

        f.write(
            f"Test samples: {len(y_test)}\n"
        )

        f.write(
            f"Test healthy: "
            f"{np.sum(y_test == 0)}\n"
        )

        f.write(
            f"Test tumor: "
            f"{np.sum(y_test == 1)}\n\n"
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        f.write(
            "SELF-SUPERVISED TRAINING\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            f"Epochs executed: "
            f"{len(history['train_loss'])}\n"
        )

        f.write(
            f"Batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Learning rate: "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"Weight decay: "
            f"{WEIGHT_DECAY}\n"
        )

        f.write(
            f"Training time: "
            f"{train_time:.2f}s\n\n"
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        f.write(
            "FEATURE REPRESENTATION\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            f"Feature dimension: "
            f"{train_features.shape[1]}\n"
        )

        f.write(
            f"Feature samples: "
            f"{train_features.shape[0]}\n\n"
        )

        # ----------------------------------------------------
        # TEST METRICS
        # ----------------------------------------------------

        f.write(
            "TEST METRICS\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            f"AUROC: "
            f"{metrics['auroc']:.4f}\n"
        )

        f.write(
            f"Average Precision: "
            f"{metrics['ap']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{metrics['f1']:.4f}\n"
        )

        f.write(
            f"Sensitivity: "
            f"{metrics['sensitivity']:.4f}\n"
        )

        f.write(
            f"Specificity: "
            f"{metrics['specificity']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy: "
            f"{metrics['bacc']:.4f}\n"
        )

        f.write(
            f"Threshold: "
            f"{metrics['threshold']:.6f}\n\n"
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70 +
            "\n"
        )

        f.write(
            f"TN: {metrics['tn']}\n"
        )

        f.write(
            f"FP: {metrics['fp']}\n"
        )

        f.write(
            f"FN: {metrics['fn']}\n"
        )

        f.write(
            f"TP: {metrics['tp']}\n"
        )

    print(
        f"✓ Saved: {path}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment():

    total_start = time.time()

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    print(
        "\n" +
        "=" * 70
    )

    print(
        " CUTPASTE — SELF-SUPERVISED "
        "ANOMALY DETECTION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Seed: {SEED}"
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print(
        "\n[1/8] Loading datasets..."
    )

    train_ds = get_dataset(
        "brats",
        img_size=IMAGE_SIZE,
        mode="train"
    )

    test_ds = get_dataset(
        "brats",
        img_size=IMAGE_SIZE,
        mode="test"
    )

    print(
        f"  ✓ Train: {len(train_ds)}"
    )

    print(
        f"  ✓ Test:  {len(test_ds)}"
    )

    # --------------------------------------------------------
    # STRUCTURE CHECK
    # --------------------------------------------------------

    train_sample = train_ds[0]

    test_sample = test_ds[0]

    print(
        "\n========== DATALOADER STRUCTURE =========="
    )

    print(
        "TRAIN keys:"
    )

    print(
        train_sample.keys()
    )

    print(
        "\nTEST keys:"
    )

    print(
        test_sample.keys()
    )

    print(
        "\nTRAIN image shape:",
        train_sample["img"].shape
    )

    print(
        "TEST image shape:",
        test_sample["img"].shape
    )

    print(
        "TEST mask shape:",
        test_sample["mask"].shape
    )

    print(
        "==========================================\n"
    )

    # ========================================================
    # 2. VERIFY TRAIN / TEST
    # ========================================================

    print(
        "\n[2/8] Verifying dataset..."
    )

    train_labels = np.asarray([
        train_ds[i]["label"]
        for i in range(
            len(train_ds)
        )
    ])

    test_labels = np.asarray([
        test_ds[i]["label"]
        for i in range(
            len(test_ds)
        )
    ])

    # --------------------------------------------------------
    # TRAIN MUST BE HEALTHY
    # --------------------------------------------------------

    if not np.all(
        train_labels == 0
    ):

        raise ValueError(
            "TRAIN contiene campioni tumorali. "
            "CutPaste deve essere addestrato "
            "esclusivamente su healthy."
        )

    print(
        "  ✓ TRAIN verified: healthy only."
    )

    print(
        "  ✓ Test labels:",
        np.unique(
            test_labels,
            return_counts=True
        )
    )

    # --------------------------------------------------------
    # IMAGE RANGE
    # --------------------------------------------------------

    train_min = min(
        train_ds[i]["img"].min().item()
        for i in range(
            len(train_ds)
        )
    )

    train_max = max(
        train_ds[i]["img"].max().item()
        for i in range(
            len(train_ds)
        )
    )

    if (
        train_min < 0.0
        or
        train_max > 1.0
    ):

        raise ValueError(
            "Le immagini devono essere "
            "nell'intervallo [0,1]."
        )

    print(
        f"  ✓ Image range: "
        f"[{train_min:.4f}, {train_max:.4f}]"
    )

    # ========================================================
    # 3. TRAIN / VALIDATION SPLIT
    # ========================================================

    print(
        "\n[3/8] Creating healthy "
        "train/validation split..."
    )

    train_indices, val_indices = (
        create_train_validation_split(
            n_samples=len(train_ds),
            val_ratio=VAL_RATIO,
            seed=SEED
        )
    )

    print(
        f"  ✓ Training healthy: "
        f"{len(train_indices)}"
    )

    print(
        f"  ✓ Validation healthy: "
        f"{len(val_indices)}"
    )

    # --------------------------------------------------------
    # SUBSETS
    # --------------------------------------------------------

    train_healthy_ds = torch.utils.data.Subset(
        train_ds,
        train_indices
    )

    val_healthy_ds = torch.utils.data.Subset(
        train_ds,
        val_indices
    )

    # ========================================================
    # 4. CUTPASTE DATASETS
    # ========================================================

    print(
        "\n[4/8] Creating self-supervised datasets..."
    )

    cutpaste_train_ds = CutPasteDataset(
        train_healthy_ds
    )

    cutpaste_val_ds = CutPasteDataset(
        val_healthy_ds
    )

    train_loader = DataLoader(
        cutpaste_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    val_loader = DataLoader(
        cutpaste_val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    print(
        f"  ✓ Self-supervised train samples: "
        f"{len(cutpaste_train_ds)}"
    )

    print(
        f"  ✓ Self-supervised validation samples: "
        f"{len(cutpaste_val_ds)}"
    )

    # ========================================================
    # 5. MODEL
    # ========================================================

    print(
        "\n[5/8] Building model..."
    )

    model = CutPasteModel().to(
        DEVICE
    )

    print(
        f"  ✓ Backbone: ResNet-18"
    )

    print(
        f"  ✓ Input: 1x{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        f"  ✓ Feature dimension: "
        f"{model.feature_dim}"
    )

    print(
        f"  ✓ Classes: {N_CLASSES}"
    )

    # ========================================================
    # 6. SELF-SUPERVISED TRAINING
    # ========================================================

    print(
        "\n[6/8] Self-supervised training..."
    )

    t0 = time.time()

    history, best_val_loss = (
        train_cutpaste(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=NUM_EPOCHS
        )
    )

    train_time = (
        time.time() - t0
    )

    print(
        f"\n✓ Best validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        f"✓ Training time: "
        f"{train_time:.2f}s"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    model_path = os.path.join(
        OUT_DIR,
        "cutpaste_best.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "feature_dim":
                model.feature_dim,

            "image_size":
                IMAGE_SIZE,

            "n_classes":
                N_CLASSES,

            "seed":
                SEED,

            "val_ratio":
                VAL_RATIO,

            "best_val_loss":
                best_val_loss
        },
        model_path
    )

    print(
        f"✓ Saved model: {model_path}"
    )

    # ========================================================
    # 7. FEATURE SPACE + THRESHOLD
    # ========================================================

    print(
        "\n[7/8] Learning healthy feature distribution..."
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    train_features, train_feature_labels = (
        extract_features(
            model,
            train_healthy_ds
        )
    )

    val_features, val_feature_labels = (
        extract_features(
            model,
            val_healthy_ds
        )
    )

    test_features, y_test = (
        extract_features(
            model,
            test_ds
        )
    )

    # --------------------------------------------------------
    # VERIFY FEATURES
    # --------------------------------------------------------

    print(
        f"  ✓ Train features: "
        f"{train_features.shape}"
    )

    print(
        f"  ✓ Validation features: "
        f"{val_features.shape}"
    )

    print(
        f"  ✓ Test features: "
        f"{test_features.shape}"
    )

    if not np.all(
        train_feature_labels == 0
    ):

        raise RuntimeError(
            "Le feature del training "
            "non sono tutte healthy."
        )

    if not np.all(
        val_feature_labels == 0
    ):

        raise RuntimeError(
            "Le feature della validation "
            "non sono tutte healthy."
        )

    # --------------------------------------------------------
    # GAUSSIAN
    # --------------------------------------------------------

    print(
        "\nFitting Gaussian density estimator..."
    )

    gde = GaussianDensityEstimator(
        regularization=GAUSSIAN_REGULARIZATION
    )

    gde.fit(
        train_features
    )

    train_scores = gde.score(
        train_features
    )

    val_scores = gde.score(
        val_features
    )

    test_scores = gde.score(
        test_features
    )

    print(
        f"  ✓ Train score range: "
        f"[{train_scores.min():.4f}, "
        f"{train_scores.max():.4f}]"
    )

    print(
        f"  ✓ Validation score range: "
        f"[{val_scores.min():.4f}, "
        f"{val_scores.max():.4f}]"
    )

    print(
        f"  ✓ Test score range: "
        f"[{test_scores.min():.4f}, "
        f"{test_scores.max():.4f}]"
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    threshold = compute_threshold(
        val_scores,
        percentile=THRESHOLD_PERCENTILE
    )

    print(
        f"\n✓ Validation P"
        f"{THRESHOLD_PERCENTILE} threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # SAVE THRESHOLD / GDE
    # --------------------------------------------------------

    density_path = os.path.join(
        OUT_DIR,
        "gaussian_density.npz"
    )

    np.savez(
        density_path,
        mean=gde.mean,
        covariance=gde.covariance,
        precision=gde.precision,
        threshold=threshold
    )

    print(
        f"✓ Saved density model: "
        f"{density_path}"
    )

    # ========================================================
    # 8. FINAL TEST EVALUATION
    # ========================================================

    print(
        "\n[8/8] Final test evaluation..."
    )

    metrics = evaluate_image_level(
        y_true=y_test,
        scores=test_scores,
        threshold=threshold
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n" +
        "=" * 70
    )

    print(
        " IMAGE-LEVEL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"AUROC:              "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"Average Precision:  "
        f"{metrics['ap']:.4f}"
    )

    print(
        f"F1:                 "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Sensitivity:        "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"Specificity:        "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"Precision:          "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Balanced Accuracy:  "
        f"{metrics['bacc']:.4f}"
    )

    print(
        f"Threshold:          "
        f"{metrics['threshold']:.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN: {metrics['tn']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )

    print(
        f"TP: {metrics['tp']}"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    print(
        "\nGenerating plots..."
    )

    plot_training_history(
        history,
        OUT_DIR
    )

    plot_results(
        metrics,
        y_test,
        OUT_DIR
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    save_metrics(
        metrics,
        OUT_DIR
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    save_report(
        metrics=metrics,
        history=history,
        train_features=train_features,
        train_indices=train_indices,
        val_indices=val_indices,
        y_test=y_test,
        out_dir=OUT_DIR,
        train_time=train_time
    )

    # ========================================================
    # FINAL
    # ========================================================

    total_time = (
        time.time()
        -
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
        "  • cutpaste_best.pt"
    )

    print(
        "  • gaussian_density.npz"
    )

    print(
        "  • training_loss.png"
    )

    print(
        "  • training_accuracy.png"
    )

    print(
        "  • score_distribution.png"
    )

    print(
        "  • confusion_matrix.png"
    )

    print(
        "  • roc_curve.png"
    )

    print(
        "  • precision_recall_curve.png"
    )

    print(
        "  • cutpaste_metrics.csv"
    )

    print(
        "  • cutpaste_report.txt"
    )

    print(
        "\nProtocol:"
    )

    print(
        "  Train:      healthy only"
    )

    print(
        "  Validation: healthy only"
    )

    print(
        "  Test:       healthy + tumor"
    )

    print(
        "  Threshold:  validation P95"
    )

    print(
        "  Seed:       42"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_experiment()

    except Exception as e:

        print(
            "\n" +
            "=" * 70
        )

        print(
            "ERROR DURING CUTPASTE EXPERIMENT"
        )

        print(
            "=" * 70
        )

        print(
            f"\n{type(e).__name__}: {e}"
        )

        raise