# ============================================================
# ISOLATION FOREST — PURE UNSUPERVISED BASELINE
# ============================================================
#
# BASELINE:
#
#   - TRAIN: NORMAL only
#   - TEST: NORMAL + TUMOR
#   - Isolation Forest
#   - contamination="auto"
#   - no cross-validation
#   - no validation set
#   - no threshold tuning on TEST
#   - native IF threshold = 0
#
# EXTENSIONS:
#
#   1. THRESHOLD ROBUSTNESS
#
#      - Native IF threshold = 0
#      - P95 of TRAIN anomaly scores
#      - P99 of TRAIN anomaly scores
#
#      P95/P99 are computed ONLY from TRAIN NORMAL scores.
#
#   2. PATIENT-LEVEL AGGREGATION
#
#      - MAX
#      - P95
#      - TOP-K MEAN
#
#      No aggregation method is selected using test labels.
#
# PRIMARY EVALUATION:
#
#   Slice-level:
#       AUROC
#       Average Precision
#
# SECONDARY / EXPLORATORY:
#
#   Slice-level threshold metrics
#   Patient-level AUROC / AP
#
# IMPORTANT:
#
#   Patient-level evaluation is exploratory because the current
#   test set contains a highly imbalanced patient distribution.
#
# ============================================================


import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    balanced_accuracy_score,
)

from torch.utils.data import DataLoader
import torch

# 1. Trova il percorso assoluto della cartella 'ml'
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Trova la cartella "padre" (la root del tuo progetto)
parent_dir = os.path.dirname(current_dir)

# 3. Aggiunge la cartella padre ai percorsi di sistema di Python
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from data_analysis import get_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 256

N_ESTIMATORS = 200

OUT_DIR = "results/isolation_forest_extended"

# ------------------------------------------------------------
# Thresholds for robustness analysis
# ------------------------------------------------------------

TRAIN_PERCENTILES = [
    95,
    99
]

# ------------------------------------------------------------
# Patient-level TOP-K aggregation
# ------------------------------------------------------------

TOP_K = 5


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PLOT STYLE
# ============================================================

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)

plt.rcParams["figure.facecolor"] = "white"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    dataset,
    batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae immagini e label.

    Input image:
        [1, 64, 64]

    Output:
        X: [N, 4096]
        y: [N]
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    X_list = []
    y_list = []

    for batch in loader:

        imgs = batch["img"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        X_batch = imgs.reshape(
            imgs.shape[0],
            -1
        )

        X_list.append(X_batch)
        y_list.append(labels)

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    return X, y


# ============================================================
# PATIENT IDS
# ============================================================

def extract_patient_ids(
    dataset
) -> np.ndarray:
    """
    Estrae il patient_id mantenendo l'ordine del dataset.
    """

    patient_ids = []

    for i in range(len(dataset)):

        sample = dataset[i]

        patient_ids.append(
            sample["patient_id"]
        )

    return np.asarray(patient_ids)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    scaler: StandardScaler = None
) -> Tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler
]:
    """
    StandardScaler fitted ONLY on training data.
    """

    if scaler is None:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

    else:

        X_train = scaler.transform(
            X_train
        )

    X_test = scaler.transform(
        X_test
    )

    return (
        X_train,
        X_test,
        scaler
    )


# ============================================================
# ISOLATION FOREST
# ============================================================

def train_iforest(
    X_train: np.ndarray
) -> IsolationForest:
    """
    Train Isolation Forest on NORMAL samples only.
    """

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )

    model.fit(
        X_train
    )

    return model


# ============================================================
# ANOMALY SCORES
# ============================================================

def get_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:

        anomaly_score:
            higher = more anomalous

        decision_score:
            native sklearn decision_function
            higher = more normal

    anomaly_score = -decision_score
    """

    decision_scores = model.decision_function(
        X
    )

    anomaly_scores = -decision_scores

    return (
        anomaly_scores,
        decision_scores
    )


# ============================================================
# THRESHOLD COMPUTATION
# ============================================================

def compute_unsupervised_thresholds(
    train_scores: np.ndarray
) -> Dict:
    """
    Computes thresholds ONLY from training scores.

    No labels are used.

    Returns:

        native:
            0.0

        p95:
            95th percentile of TRAIN anomaly scores

        p99:
            99th percentile of TRAIN anomaly scores
    """

    thresholds = {

        "native_0":
            0.0,

        "train_p95":
            float(
                np.percentile(
                    train_scores,
                    95
                )
            ),

        "train_p99":
            float(
                np.percentile(
                    train_scores,
                    99
                )
            )
    }

    return thresholds


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float
) -> Dict:
    """
    Binary classification using a fixed anomaly threshold.

        score > threshold -> anomaly
        score <= threshold -> normal

    Threshold must be defined independently from test labels.
    """

    y_pred = (
        scores > threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp /
        (tp + fn + 1e-8)
    )

    specificity = (
        tn /
        (tn + fp + 1e-8)
    )

    precision_value = (
        tp /
        (tp + fp + 1e-8)
    )

    f1 = (
        2 *
        precision_value *
        sensitivity /
        (
            precision_value +
            sensitivity +
            1e-8
        )
    )

    bacc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    return {

        "threshold":
            threshold,

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

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision_value,

        "f1":
            f1,

        "bacc":
            bacc
    }


# ============================================================
# SLICE-LEVEL RANKING METRICS
# ============================================================

def evaluate_ranking(
    scores: np.ndarray,
    y_true: np.ndarray
) -> Dict:
    """
    Threshold-independent evaluation.

    AUROC and Average Precision.
    """

    auroc = roc_auc_score(
        y_true,
        scores
    )

    ap = average_precision_score(
        y_true,
        scores
    )

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        scores
    )

    (
        precision,
        recall,
        pr_thresholds
    ) = precision_recall_curve(
        y_true,
        scores
    )

    return {

        "auroc":
            auroc,

        "ap":
            ap,

        "fpr":
            fpr,

        "tpr":
            tpr,

        "roc_thresholds":
            roc_thresholds,

        "precision_curve":
            precision,

        "recall_curve":
            recall,

        "pr_thresholds":
            pr_thresholds
    }


# ============================================================
# SLICE-LEVEL BASELINE EVALUATION
# ============================================================

def evaluate_slice_level(
    model: IsolationForest,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Complete slice-level evaluation.

    The native threshold is 0.

    Additional thresholds are computed ONLY from
    TRAIN NORMAL anomaly scores.
    """

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    train_scores, train_decision_scores = (
        get_anomaly_scores(
            model,
            X_train
        )
    )

    test_scores, test_decision_scores = (
        get_anomaly_scores(
            model,
            X_test
        )
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = compute_unsupervised_thresholds(
        train_scores
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    ranking = evaluate_ranking(
        test_scores,
        y_test
    )

    # --------------------------------------------------------
    # Threshold metrics
    # --------------------------------------------------------

    threshold_results = {}

    for name, threshold in thresholds.items():

        threshold_results[name] = evaluate_threshold(
            scores=test_scores,
            y_true=y_test,
            threshold=threshold
        )

    return {

        "train_scores":
            train_scores,

        "train_decision_scores":
            train_decision_scores,

        "scores":
            test_scores,

        "decision_scores":
            test_decision_scores,

        "thresholds":
            thresholds,

        "ranking":
            ranking,

        "threshold_results":
            threshold_results
    }


# ============================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================

def aggregate_patient_scores(
    scores: np.ndarray,
    y_test: np.ndarray,
    patient_ids: np.ndarray,
    top_k: int = TOP_K
) -> Dict:
    """
    Aggregates slice anomaly scores at patient level.

    Aggregations:

        MAX
        P95
        TOP-K MEAN

    Ground truth patient label:

        tumor if at least one slice is tumor.
    """

    unique_patients = np.unique(
        patient_ids
    )

    patient_ids_out = []

    patient_labels = []

    max_scores = []
    p95_scores = []
    topk_scores = []

    for patient_id in unique_patients:

        mask = (
            patient_ids == patient_id
        )

        slice_scores = scores[
            mask
        ]

        slice_labels = y_test[
            mask
        ]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        patient_label = int(
            np.any(
                slice_labels == 1
            )
        )

        # ----------------------------------------------------
        # MAX
        # ----------------------------------------------------

        max_score = np.max(
            slice_scores
        )

        # ----------------------------------------------------
        # P95
        # ----------------------------------------------------

        p95_score = np.percentile(
            slice_scores,
            95
        )

        # ----------------------------------------------------
        # TOP-K MEAN
        # ----------------------------------------------------

        k = min(
            top_k,
            len(slice_scores)
        )

        top_scores = np.sort(
            slice_scores
        )[-k:]

        topk_mean = np.mean(
            top_scores
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        patient_ids_out.append(
            patient_id
        )

        patient_labels.append(
            patient_label
        )

        max_scores.append(
            max_score
        )

        p95_scores.append(
            p95_score
        )

        topk_scores.append(
            topk_mean
        )

    return {

        "patient_ids":
            np.asarray(patient_ids_out),

        "labels":
            np.asarray(patient_labels),

        "max":
            np.asarray(max_scores),

        "p95":
            np.asarray(p95_scores),

        "topk_mean":
            np.asarray(topk_scores)
    }


# ============================================================
# PATIENT-LEVEL METRICS
# ============================================================

def evaluate_patient_aggregation(
    patient_results: Dict
) -> Dict:
    """
    Computes AUROC and AP for each aggregation strategy.

    No threshold tuning.
    """

    labels = patient_results["labels"]

    results = {}

    aggregation_names = [
        "max",
        "p95",
        "topk_mean"
    ]

    for aggregation_name in aggregation_names:

        scores = patient_results[
            aggregation_name
        ]

        auroc = roc_auc_score(
            labels,
            scores
        )

        ap = average_precision_score(
            labels,
            scores
        )

        results[aggregation_name] = {

            "auroc":
                auroc,

            "ap":
                ap,

            "scores":
                scores
        }

    return results


# ============================================================
# SAVE THRESHOLD ANALYSIS
# ============================================================

def save_threshold_analysis(
    slice_results: Dict,
    out_dir: str
) -> None:
    """
    Saves threshold robustness analysis.

    All thresholds are derived without test labels.
    """

    rows = []

    for name, result in (
        slice_results["threshold_results"].items()
    ):

        rows.append({

            "threshold_method":
                name,

            "threshold":
                result["threshold"],

            "Sensitivity":
                result["sensitivity"],

            "Specificity":
                result["specificity"],

            "Precision":
                result["precision"],

            "F1":
                result["f1"],

            "Balanced_Accuracy":
                result["bacc"],

            "TN":
                result["tn"],

            "FP":
                result["fp"],

            "FN":
                result["fn"],

            "TP":
                result["tp"]
        })

    df = pd.DataFrame(
        rows
    )

    output_path = os.path.join(
        out_dir,
        "threshold_analysis.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# SAVE PATIENT ANALYSIS
# ============================================================

def save_patient_analysis(
    patient_results: Dict,
    patient_metrics: Dict,
    out_dir: str
) -> None:
    """
    Saves patient-level scores and metrics.
    """

    # --------------------------------------------------------
    # Patient scores
    # --------------------------------------------------------

    df_scores = pd.DataFrame({

        "patient_id":
            patient_results["patient_ids"],

        "true_label":
            patient_results["labels"],

        "max_score":
            patient_results["max"],

        "p95_score":
            patient_results["p95"],

        "top5_mean_score":
            patient_results["topk_mean"]
    })

    score_path = os.path.join(
        out_dir,
        "patient_level_results.csv"
    )

    df_scores.to_csv(
        score_path,
        index=False
    )

    print(
        f"  ✓ Saved: {score_path}"
    )

    # --------------------------------------------------------
    # Patient metrics
    # --------------------------------------------------------

    rows = []

    for aggregation, values in (
        patient_metrics.items()
    ):

        rows.append({

            "aggregation":
                aggregation,

            "AUROC":
                values["auroc"],

            "Average_Precision":
                values["ap"]
        })

    df_metrics = pd.DataFrame(
        rows
    )

    metrics_path = os.path.join(
        out_dir,
        "patient_level_metrics.csv"
    )

    df_metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"  ✓ Saved: {metrics_path}"
    )


# ============================================================
# SAVE SLICE ERROR ANALYSIS
# ============================================================

def save_error_analysis(
    y_test: np.ndarray,
    scores: np.ndarray,
    decision_scores: np.ndarray,
    patient_ids: np.ndarray,
    native_threshold_result: Dict,
    out_dir: str
) -> None:
    """
    Saves TN / FP / FN / TP using the NATIVE IF threshold.

    This preserves the original baseline evaluation.
    """

    y_pred = native_threshold_result[
        "y_pred"
    ]

    all_indices = np.arange(
        len(y_test)
    )

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
            all_indices,

        "patient_id":
            patient_ids,

        "true_label":
            y_test,

        "predicted_label":
            y_pred,

        "category":
            categories,

        "anomaly_score":
            scores,

        "decision_score":
            decision_scores
    })

    output_path = os.path.join(
        out_dir,
        "error_analysis.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# PLOT MAIN RESULTS
# ============================================================

def plot_main_results(
    slice_results: Dict,
    y_test: np.ndarray,
    out_dir: str
) -> None:
    """
    Main 2x2 figure.

    A: score distribution
    B: native confusion matrix
    C: ROC
    D: PR
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 12)
    )

    scores = slice_results["scores"]

    ranking = slice_results["ranking"]

    native = slice_results[
        "threshold_results"
    ]["native_0"]

    thresholds = slice_results[
        "thresholds"
    ]

    # ========================================================
    # A — SCORE DISTRIBUTION
    # ========================================================

    ax = axes[0, 0]

    if np.sum(y_test == 0) > 1:

        sns.kdeplot(
            scores[y_test == 0],
            fill=True,
            color="#2ca02c",
            alpha=0.4,
            label="Healthy",
            ax=ax,
            linewidth=2
        )

    if np.sum(y_test == 1) > 1:

        sns.kdeplot(
            scores[y_test == 1],
            fill=True,
            color="#d62728",
            alpha=0.4,
            label="Tumor",
            ax=ax,
            linewidth=2
        )

    # Native threshold

    ax.axvline(
        thresholds["native_0"],
        linestyle="--",
        color="black",
        linewidth=2,
        label="Native IF threshold = 0"
    )

    # P95

    ax.axvline(
        thresholds["train_p95"],
        linestyle=":",
        color="blue",
        linewidth=2,
        label="Train P95"
    )

    # P99

    ax.axvline(
        thresholds["train_p99"],
        linestyle=":",
        color="purple",
        linewidth=2,
        label="Train P99"
    )

    ax.set_xlabel(
        "Anomaly Score"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "(A) Anomaly Score Distribution",
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    # ========================================================
    # B — CONFUSION MATRIX
    # ========================================================

    ax = axes[0, 1]

    sns.heatmap(
        native["cm"],
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=False,
        xticklabels=[
            "Healthy",
            "Tumor"
        ],
        yticklabels=[
            "Healthy",
            "Tumor"
        ],
        annot_kws={
            "fontsize": 11,
            "fontweight": "bold"
        }
    )

    ax.set_ylabel(
        "True Label"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_title(
        "(B) Native IF Confusion Matrix",
        fontweight="bold"
    )

    # ========================================================
    # C — ROC
    # ========================================================

    ax = axes[1, 0]

    ax.plot(
        ranking["fpr"],
        ranking["tpr"],
        linewidth=2.5,
        label=(
            f"AUROC = "
            f"{ranking['auroc']:.3f}"
        )
    )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        linewidth=1.5,
        alpha=0.5
    )

    ax.fill_between(
        ranking["fpr"],
        ranking["tpr"],
        alpha=0.2
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

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    # ========================================================
    # D — PRECISION RECALL
    # ========================================================

    ax = axes[1, 1]

    baseline = np.mean(
        y_test
    )

    ax.plot(
        ranking["recall_curve"],
        ranking["precision_curve"],
        linewidth=2.5,
        label=(
            f"AP = "
            f"{ranking['ap']:.3f}"
        )
    )

    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        alpha=0.5,
        label=(
            f"Baseline = "
            f"{baseline:.3f}"
        )
    )

    ax.fill_between(
        ranking["recall_curve"],
        ranking["precision_curve"],
        alpha=0.2
    )

    ax.set_xlabel(
        "Recall (Sensitivity)"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "(D) Precision-Recall Curve",
        fontweight="bold"
    )

    ax.legend(
        loc="best"
    )

    ax.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        out_dir,
        "isolation_forest_results.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# PLOT THRESHOLD COMPARISON
# ============================================================

def plot_threshold_comparison(
    slice_results: Dict,
    out_dir: str
) -> None:
    """
    Visual comparison of threshold-dependent metrics.
    """

    names = []
    sensitivity = []
    specificity = []
    f1 = []
    bacc = []

    for name, result in (
        slice_results[
            "threshold_results"
        ].items()
    ):

        names.append(
            name
        )

        sensitivity.append(
            result["sensitivity"]
        )

        specificity.append(
            result["specificity"]
        )

        f1.append(
            result["f1"]
        )

        bacc.append(
            result["bacc"]
        )

    x = np.arange(
        len(names)
    )

    width = 0.2

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - 1.5 * width,
        sensitivity,
        width,
        label="Sensitivity"
    )

    ax.bar(
        x - 0.5 * width,
        specificity,
        width,
        label="Specificity"
    )

    ax.bar(
        x + 0.5 * width,
        f1,
        width,
        label="F1"
    )

    ax.bar(
        x + 1.5 * width,
        bacc,
        width,
        label="Balanced Accuracy"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        names
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Threshold Robustness Analysis",
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        out_dir,
        "threshold_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# PLOT PATIENT AGGREGATION
# ============================================================

def plot_patient_aggregation(
    patient_metrics: Dict,
    out_dir: str
) -> None:
    """
    Compares patient-level aggregation strategies.
    """

    names = list(
        patient_metrics.keys()
    )

    aurocs = [
        patient_metrics[name]["auroc"]
        for name in names
    ]

    aps = [
        patient_metrics[name]["ap"]
        for name in names
    ]

    x = np.arange(
        len(names)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        aurocs,
        width,
        label="AUROC"
    )

    ax.bar(
        x + width / 2,
        aps,
        width,
        label="Average Precision"
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color="gray",
        alpha=0.5,
        label="AUROC chance level"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "MAX",
            "P95",
            "TOP-5 MEAN"
        ]
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Patient-Level Aggregation Sensitivity Analysis",
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = os.path.join(
        out_dir,
        "patient_aggregation_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# SAVE COMPLETE METRICS
# ============================================================

def save_metrics_csv(
    slice_results: Dict,
    patient_results: Dict,
    patient_metrics: Dict,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_time: float,
    out_dir: str
) -> None:
    """
    Saves a compact summary of all experiment results.
    """

    ranking = slice_results[
        "ranking"
    ]

    native = slice_results[
        "threshold_results"
    ]["native_0"]

    thresholds = slice_results[
        "thresholds"
    ]

    rows = {

        # ----------------------------------------------------
        # Primary slice-level metrics
        # ----------------------------------------------------

        "Slice_AUROC":
            ranking["auroc"],

        "Slice_Average_Precision":
            ranking["ap"],

        # ----------------------------------------------------
        # Native IF threshold
        # ----------------------------------------------------

        "Native_Threshold":
            thresholds["native_0"],

        "Native_Sensitivity":
            native["sensitivity"],

        "Native_Specificity":
            native["specificity"],

        "Native_F1":
            native["f1"],

        "Native_Balanced_Accuracy":
            native["bacc"],

        "Native_TN":
            native["tn"],

        "Native_FP":
            native["fp"],

        "Native_FN":
            native["fn"],

        "Native_TP":
            native["tp"],

        # ----------------------------------------------------
        # Unsupervised thresholds
        # ----------------------------------------------------

        "Train_P95_Threshold":
            thresholds["train_p95"],

        "Train_P99_Threshold":
            thresholds["train_p99"],

        # ----------------------------------------------------
        # Patient-level
        # ----------------------------------------------------

        "Patient_MAX_AUROC":
            patient_metrics["max"]["auroc"],

        "Patient_MAX_AP":
            patient_metrics["max"]["ap"],

        "Patient_P95_AUROC":
            patient_metrics["p95"]["auroc"],

        "Patient_P95_AP":
            patient_metrics["p95"]["ap"],

        "Patient_TOP5_MEAN_AUROC":
            patient_metrics["topk_mean"]["auroc"],

        "Patient_TOP5_MEAN_AP":
            patient_metrics["topk_mean"]["ap"],

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        "N_Train":
            len(y_train),

        "N_Test":
            len(y_test),

        "N_Train_Healthy":
            int(
                np.sum(
                    y_train == 0
                )
            ),

        "N_Test_Healthy":
            int(
                np.sum(
                    y_test == 0
                )
            ),

        "N_Test_Tumor":
            int(
                np.sum(
                    y_test == 1
                )
            ),

        "N_Patients":
            len(
                patient_results["labels"]
            ),

        "N_Healthy_Patients":
            int(
                np.sum(
                    patient_results["labels"] == 0
                )
            ),

        "N_Tumor_Patients":
            int(
                np.sum(
                    patient_results["labels"] == 1
                )
            ),

        "Training_Time_Seconds":
            train_time
    }

    df = pd.DataFrame(
        [rows]
    )

    output_path = os.path.join(
        out_dir,
        "isolation_forest_metrics.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# REPORT
# ============================================================

def save_report(
    slice_results: Dict,
    patient_results: Dict,
    patient_metrics: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_time: float,
    out_dir: str
) -> None:
    """
    Saves complete textual report.
    """

    report_path = os.path.join(
        out_dir,
        "isolation_forest_report.txt"
    )

    ranking = slice_results[
        "ranking"
    ]

    thresholds = slice_results[
        "thresholds"
    ]

    threshold_results = slice_results[
        "threshold_results"
    ]

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "ISOLATION FOREST — PURE UNSUPERVISED BASELINE\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        # ====================================================
        # METHODOLOGY
        # ====================================================

        f.write(
            "METHODOLOGY\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Training performed exclusively on healthy samples.\n"
        )

        f.write(
            "Tumor labels were NOT used during training.\n"
        )

        f.write(
            "No cross-validation was performed.\n"
        )

        f.write(
            "No validation set was used.\n"
        )

        f.write(
            "No threshold tuning was performed on the test set.\n"
        )

        f.write(
            "The native Isolation Forest threshold was evaluated.\n"
        )

        f.write(
            "Additional thresholds were derived exclusively from\n"
        )

        f.write(
            "the anomaly-score distribution of the healthy training set.\n\n"
        )

        # ====================================================
        # HYPERPARAMETERS
        # ====================================================

        f.write(
            "HYPERPARAMETERS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"n_estimators: {N_ESTIMATORS}\n"
        )

        f.write(
            "contamination: auto\n"
        )

        f.write(
            f"random_state: {SEED}\n"
        )

        f.write(
            "n_jobs: -1\n\n"
        )

        # ====================================================
        # FEATURES
        # ====================================================

        f.write(
            "FEATURE ENGINEERING\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Training shape: {X_train.shape}\n"
        )

        f.write(
            f"Feature dimension: {X_train.shape[1]}\n"
        )

        f.write(
            "Input representation: flattened 64x64 image\n"
        )

        f.write(
            "Normalization: StandardScaler\n"
        )

        f.write(
            "Scaler fitted exclusively on training data.\n\n"
        )

        # ====================================================
        # TRAINING
        # ====================================================

        f.write(
            "TRAINING\n"
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
            f"Train tumor: {np.sum(y_train == 1)}\n"
        )

        f.write(
            f"Training time: {train_time:.2f}s\n"
        )

        f.write(
            "Learning paradigm: unsupervised anomaly detection.\n\n"
        )

        # ====================================================
        # TEST
        # ====================================================

        f.write(
            "TEST SET\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Test samples: {len(y_test)}\n"
        )

        f.write(
            f"Test healthy: {np.sum(y_test == 0)}\n"
        )

        f.write(
            f"Test tumor: {np.sum(y_test == 1)}\n"
        )

        f.write(
            f"Tumor proportion: {np.mean(y_test):.2%}\n\n"
        )

        # ====================================================
        # PRIMARY SLICE RESULTS
        # ====================================================

        f.write(
            "PRIMARY SLICE-LEVEL RESULTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"AUROC:             "
            f"{ranking['auroc']:.4f}\n"
        )

        f.write(
            f"Average Precision: "
            f"{ranking['ap']:.4f}\n\n"
        )

        # ====================================================
        # THRESHOLD ANALYSIS
        # ====================================================

        f.write(
            "THRESHOLD ROBUSTNESS ANALYSIS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Thresholds P95/P99 are computed exclusively from\n"
        )

        f.write(
            "healthy training anomaly scores.\n\n"
        )

        for name, result in (
            threshold_results.items()
        ):

            f.write(
                f"{name}\n"
            )

            f.write(
                f"  Threshold: "
                f"{result['threshold']:.6f}\n"
            )

            f.write(
                f"  Sensitivity: "
                f"{result['sensitivity']:.4f}\n"
            )

            f.write(
                f"  Specificity: "
                f"{result['specificity']:.4f}\n"
            )

            f.write(
                f"  Precision: "
                f"{result['precision']:.4f}\n"
            )

            f.write(
                f"  F1: "
                f"{result['f1']:.4f}\n"
            )

            f.write(
                f"  Balanced Accuracy: "
                f"{result['bacc']:.4f}\n"
            )

            f.write(
                "\n"
            )

        # ====================================================
        # PATIENT LEVEL
        # ====================================================

        f.write(
            "PATIENT-LEVEL ANALYSIS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Patients: "
            f"{len(patient_results['labels'])}\n"
        )

        f.write(
            f"Healthy patients: "
            f"{np.sum(patient_results['labels'] == 0)}\n"
        )

        f.write(
            f"Tumor patients: "
            f"{np.sum(patient_results['labels'] == 1)}\n"
        )

        f.write(
            "Patient ground truth: tumor if at least one slice is tumor.\n"
        )

        f.write(
            "Patient aggregation strategies:\n"
        )

        f.write(
            "  - MAX\n"
        )

        f.write(
            "  - P95\n"
        )

        f.write(
            f"  - TOP-{TOP_K} MEAN\n\n"
        )

        for aggregation, result in (
            patient_metrics.items()
        ):

            f.write(
                f"{aggregation}\n"
            )

            f.write(
                f"  AUROC: "
                f"{result['auroc']:.4f}\n"
            )

            f.write(
                f"  Average Precision: "
                f"{result['ap']:.4f}\n\n"
            )

        f.write(
            "IMPORTANT:\n"
        )

        f.write(
            "Patient-level results are considered exploratory because\n"
        )

        f.write(
            "the current test set contains a highly imbalanced patient\n"
        )

        f.write(
            "distribution.\n\n"
        )

        # ====================================================
        # INTERPRETATION
        # ====================================================

        f.write(
            "INTERPRETATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "The Isolation Forest was trained exclusively on healthy\n"
        )

        f.write(
            "samples without access to tumor labels.\n"
        )

        f.write(
            "The primary evaluation is based on threshold-independent\n"
        )

        f.write(
            "slice-level AUROC and Average Precision.\n"
        )

        f.write(
            "The native IF threshold is reported as the primary binary\n"
        )

        f.write(
            "decision rule.\n"
        )

        f.write(
            "P95 and P99 thresholds are reported only as unsupervised\n"
        )

        f.write(
            "robustness analyses because they are derived exclusively\n"
        )

        f.write(
            "from the healthy training score distribution.\n"
        )

        f.write(
            "Patient-level aggregation is treated as a secondary\n"
        )

        f.write(
            "sensitivity analysis rather than the primary endpoint.\n\n"
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
        f"  ✓ Saved: {report_path}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment() -> None:

    t_total = time.time()

    print(
        "\n" +
        "=" * 70
    )

    print(
        " ISOLATION FOREST — PURE UNSUPERVISED BASELINE"
    )

    print(
        "=" * 70
    )

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    # ========================================================
    # 1. DATA LOADING
    # ========================================================

    print(
        "\n[1/5] Loading datasets..."
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
        f"  ✓ Train: {len(train_ds)} samples"
    )

    print(
        f"  ✓ Test:  {len(test_ds)} samples"
    )

    print(
        f"  Load time: "
        f"{time.time() - t0:.2f}s"
    )

    # ========================================================
    # 2. FEATURE EXTRACTION
    # ========================================================

    print(
        "\n[2/5] Feature extraction..."
    )

    t0 = time.time()

    X_train, y_train = extract_features(
        train_ds
    )

    X_test, y_test = extract_features(
        test_ds
    )

    patient_ids_test = extract_patient_ids(
        test_ds
    )

    print(
        f"  ✓ X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"  ✓ X_test shape:  "
        f"{X_test.shape}"
    )

    print(
        f"  ✓ Train labels: "
        f"{np.unique(y_train, return_counts=True)}"
    )

    print(
        f"  ✓ Test labels: "
        f"{np.unique(y_test, return_counts=True)}"
    )

    print(
        f"  Feature extraction time: "
        f"{time.time() - t0:.2f}s"
    )

    # ========================================================
    # VERIFY TRAIN
    # ========================================================

    if not np.all(
        y_train == 0
    ):

        raise ValueError(
            "ERROR: training set contains tumor samples. "
            "Pure unsupervised baseline requires NORMAL only."
        )

    print(
        "  ✓ Verified: TRAIN contains NORMAL samples only."
    )

    # ========================================================
    # 3. NORMALIZATION
    # ========================================================

    print(
        "\n[3/5] Feature normalization..."
    )

    t0 = time.time()

    X_train, X_test, scaler = normalize_features(
        X_train,
        X_test
    )

    print(
        f"  ✓ X_train mean: "
        f"{X_train.mean():.4f}"
    )

    print(
        f"  ✓ X_train std: "
        f"{X_train.std():.4f}"
    )

    print(
        f"  Normalization time: "
        f"{time.time() - t0:.2f}s"
    )

    # ========================================================
    # 4. TRAINING
    # ========================================================

    print(
        "\n[4/5] Training Isolation Forest..."
    )

    print(
        "  Training uses NORMAL samples only."
    )

    print(
        "  contamination = auto"
    )

    print(
        "  Native threshold = 0"
    )

    print(
        "  No test-set threshold tuning."
    )

    t0 = time.time()

    model = train_iforest(
        X_train
    )

    train_time = (
        time.time() - t0
    )

    print(
        f"  ✓ Model trained in "
        f"{train_time:.2f}s"
    )

    # ========================================================
    # 5. EVALUATION
    # ========================================================

    print(
        "\n[5/5] Evaluation..."
    )

    t0 = time.time()

    slice_results = evaluate_slice_level(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test
    )

    # ========================================================
    # PRINT PRIMARY RESULTS
    # ========================================================

    ranking = slice_results[
        "ranking"
    ]

    print(
        "\n  PRIMARY SLICE-LEVEL RESULTS"
    )

    print(
        f"  AUROC: "
        f"{ranking['auroc']:.4f}"
    )

    print(
        f"  Average Precision: "
        f"{ranking['ap']:.4f}"
    )

    # ========================================================
    # PRINT THRESHOLDS
    # ========================================================

    print(
        "\n  UNSUPERVISED THRESHOLDS"
    )

    for name, threshold in (
        slice_results["thresholds"].items()
    ):

        print(
            f"  {name}: "
            f"{threshold:.6f}"
        )

    # ========================================================
    # PRINT THRESHOLD RESULTS
    # ========================================================

    print(
        "\n  THRESHOLD ANALYSIS"
    )

    for name, result in (
        slice_results[
            "threshold_results"
        ].items()
    ):

        print(
            f"\n  {name}"
        )

        print(
            f"    Sensitivity: "
            f"{result['sensitivity']:.4f}"
        )

        print(
            f"    Specificity: "
            f"{result['specificity']:.4f}"
        )

        print(
            f"    F1: "
            f"{result['f1']:.4f}"
        )

        print(
            f"    Balanced Accuracy: "
            f"{result['bacc']:.4f}"
        )

        print(
            f"    TN: {result['tn']}"
        )

        print(
            f"    FP: {result['fp']}"
        )

        print(
            f"    FN: {result['fn']}"
        )

        print(
            f"    TP: {result['tp']}"
        )

    # ========================================================
    # PATIENT LEVEL
    # ========================================================

    patient_results = aggregate_patient_scores(
        scores=slice_results["scores"],
        y_test=y_test,
        patient_ids=patient_ids_test,
        top_k=TOP_K
    )

    patient_metrics = evaluate_patient_aggregation(
        patient_results
    )

    print(
        "\n  PATIENT-LEVEL RESULTS"
    )

    print(
        f"  Patients: "
        f"{len(patient_results['labels'])}"
    )

    print(
        f"  Healthy patients: "
        f"{np.sum(patient_results['labels'] == 0)}"
    )

    print(
        f"  Tumor patients: "
        f"{np.sum(patient_results['labels'] == 1)}"
    )

    for aggregation, result in (
        patient_metrics.items()
    ):

        print(
            f"\n  {aggregation}"
        )

        print(
            f"    AUROC: "
            f"{result['auroc']:.4f}"
        )

        print(
            f"    Average Precision: "
            f"{result['ap']:.4f}"
        )

    eval_time = (
        time.time() - t0
    )

    print(
        f"\n  Evaluation time: "
        f"{eval_time:.2f}s"
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    print(
        "\n[VISUALIZATION] Generating plots..."
    )

    plot_main_results(
        slice_results,
        y_test,
        OUT_DIR
    )

    plot_threshold_comparison(
        slice_results,
        OUT_DIR
    )

    plot_patient_aggregation(
        patient_metrics,
        OUT_DIR
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    print(
        "\n[ERROR ANALYSIS] Saving native IF predictions..."
    )

    save_error_analysis(
        y_test=y_test,
        scores=slice_results["scores"],
        decision_scores=slice_results["decision_scores"],
        patient_ids=patient_ids_test,
        native_threshold_result=
            slice_results[
                "threshold_results"
            ]["native_0"],
        out_dir=OUT_DIR
    )

    # ========================================================
    # THRESHOLD ANALYSIS CSV
    # ========================================================

    print(
        "\n[RESULTS] Saving threshold analysis..."
    )

    save_threshold_analysis(
        slice_results,
        OUT_DIR
    )

    # ========================================================
    # PATIENT ANALYSIS CSV
    # ========================================================

    print(
        "\n[PATIENT RESULTS] Saving patient analysis..."
    )

    save_patient_analysis(
        patient_results,
        patient_metrics,
        OUT_DIR
    )

    # ========================================================
    # COMPLETE METRICS
    # ========================================================

    print(
        "\n[RESULTS] Saving numerical metrics..."
    )

    save_metrics_csv(
        slice_results=slice_results,
        patient_results=patient_results,
        patient_metrics=patient_metrics,
        y_train=y_train,
        y_test=y_test,
        train_time=train_time,
        out_dir=OUT_DIR
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n[REPORTING] Saving report..."
    )

    save_report(
        slice_results=slice_results,
        patient_results=patient_results,
        patient_metrics=patient_metrics,
        X_train=X_train,
        y_train=y_train,
        y_test=y_test,
        train_time=train_time,
        out_dir=OUT_DIR
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_time = (
        time.time() -
        t_total
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
        "  • isolation_forest_results.png"
    )

    print(
        "  • threshold_comparison.png"
    )

    print(
        "  • patient_aggregation_comparison.png"
    )

    print(
        "  • isolation_forest_metrics.csv"
    )

    print(
        "  • threshold_analysis.csv"
    )

    print(
        "  • patient_level_results.csv"
    )

    print(
        "  • patient_level_metrics.csv"
    )

    print(
        "  • error_analysis.csv"
    )

    print(
        "  • isolation_forest_report.txt"
    )

    print(
        "\nMethodology:"
    )

    print(
        "  • Training: NORMAL only"
    )

    print(
        "  • Test: NORMAL + TUMOR"
    )

    print(
        "  • contamination: auto"
    )

    print(
        "  • Native threshold: 0"
    )

    print(
        "  • Additional thresholds: TRAIN P95 / TRAIN P99"
    )

    print(
        "  • Threshold tuning on test: NO"
    )

    print(
        "  • Cross-validation: NO"
    )

    print(
        "  • Validation set: NO"
    )

    print(
        "  • Patient aggregation: MAX / P95 / TOP-5 MEAN"
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