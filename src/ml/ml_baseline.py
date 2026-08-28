# ============================================================
# ISOLATION FOREST — PURE UNSUPERVISED BASELINE
# ============================================================
#
# Baseline metodologica:
#
#   - TRAIN: esclusivamente immagini NORMAL
#   - TEST: NORMAL + TUMOR
#   - Isolation Forest standard
#   - contamination="auto"
#   - nessuna cross-validation
#   - nessun validation set
#   - nessun threshold tuning
#   - soglia decisionale naturale di Isolation Forest = 0
#   - valutazione esclusivamente SLICE-LEVEL
#
# Obiettivo:
#
# valutare quanto un modello addestrato esclusivamente
# su immagini normali riesca a identificare le immagini
# tumorali come anomalie.
#
# IMPORTANTE:
#
# Questa implementazione rappresenta una BASELINE PURA.
# Non vengono utilizzate:
#
#   - label tumorali durante il training
#   - soglie adattive
#   - percentile threshold
#   - threshold tuning
#   - aggregazioni patient-level
#   - metriche patient-level
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
from data_analysis.dataloader import get_dataset


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 256

N_ESTIMATORS = 200

OUT_DIR = "results/ml_baseline"


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
    Estrae immagini e label dal dataset.

    Input:
        dataset: BraTSDataset

    Output:
        X: feature matrix [N, D]
        y: labels [N]

    Le immagini vengono appiattite:

        [1, 64, 64] -> [4096]
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    X_list = []
    y_list = []
    pid_list = []

    for batch in loader:

        imgs = batch["img"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        # Gestisce il recupero del patient_id se presente nel batch, altrimenti mette stringhe vuote o indici
        if "patient_id" in batch:
            pids = batch["patient_id"]
            if isinstance(pids, torch.Tensor):
                pids = pids.cpu().numpy()
        else:
            # Fallback se il dataloader non restituisce direttamente i patient_id
            pids = np.array(["unknown"] * len(labels))

        X_batch = imgs.reshape(
            imgs.shape[0],
            -1
        )

        X_list.append(X_batch)
        y_list.append(labels)
        pid_list.append(pids)

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    pids = np.concatenate(pid_list, axis=0)

    return X, y, pids


# ============================================================
# FEATURE NORMALIZATION
# ============================================================

def normalize_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[
    np.ndarray,
    np.ndarray,
    StandardScaler
]:
    """
    Standardizza le feature.

    Lo scaler viene FITTATO esclusivamente sul TRAIN.

    Il TEST viene solamente trasformato.

    Questo evita data leakage.
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
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
    Addestra Isolation Forest esclusivamente
    sulle immagini NORMAL.

    contamination="auto":

        non viene fornita alcuna stima manuale
        della proporzione di anomalie.

    Le label tumorali NON vengono utilizzate.
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
# SLICE-LEVEL EVALUATION
# ============================================================

def evaluate_iforest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Valuta Isolation Forest sul test set.

    Isolation Forest sklearn:

        decision_function > 0 -> NORMAL
        decision_function < 0 -> ANOMALY

    Per ottenere uno score in cui:

        score alto -> maggiore anomalia

    viene utilizzato:

        anomaly_score = -decision_function

    La soglia naturale diventa quindi:

        anomaly_score = 0

    Non viene effettuato alcun threshold tuning.
    """

    # ========================================================
    # DECISION FUNCTION
    # ========================================================

    decision_scores = model.decision_function(
        X_test
    )

    # Score orientato verso l'anomalia:
    #
    # alto  -> più anomalo
    # basso -> più normale

    anomaly_scores = -decision_scores

    # ========================================================
    # NATURAL ISOLATION FOREST THRESHOLD
    # ========================================================

    threshold = 0.0

    # ========================================================
    # PREDICTIONS
    # ========================================================

    # sklearn:
    #
    #   +1 = normal
    #   -1 = anomaly

    y_pred_raw = model.predict(
        X_test
    )

    y_pred = (
        y_pred_raw == -1
    ).astype(int)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # ========================================================
    # AUROC
    # ========================================================

    auroc = roc_auc_score(
        y_test,
        anomaly_scores
    )

    # ========================================================
    # AVERAGE PRECISION
    # ========================================================

    ap = average_precision_score(
        y_test,
        anomaly_scores
    )

    # ========================================================
    # SENSITIVITY
    # ========================================================

    sensitivity = (
        tp /
        (tp + fn + 1e-8)
    )

    # ========================================================
    # SPECIFICITY
    # ========================================================

    specificity = (
        tn /
        (tn + fp + 1e-8)
    )

    # ========================================================
    # PRECISION
    # ========================================================

    precision_value = (
        tp /
        (tp + fp + 1e-8)
    )

    # ========================================================
    # F1
    # ========================================================

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

    # ========================================================
    # BALANCED ACCURACY
    # ========================================================

    bacc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # ========================================================
    # ROC CURVE
    # ========================================================

    fpr, tpr, roc_thresholds = roc_curve(
        y_test,
        anomaly_scores
    )

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    (
        precision_curve,
        recall_curve,
        pr_thresholds
    ) = precision_recall_curve(
        y_test,
        anomaly_scores
    )

    return {

        # Metrics
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_value": precision_value,
        "bacc": bacc,

        # Natural threshold
        "threshold": threshold,

        # Scores
        "scores": anomaly_scores,
        "decision_scores": decision_scores,

        # Predictions
        "y_pred": y_pred,

        # Confusion matrix
        "cm": cm,

        # ROC
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,

        # Precision-Recall
        "precision": precision_curve,
        "recall": recall_curve,
        "pr_thresholds": pr_thresholds,

        # Confusion matrix values
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


# ============================================================
# MAIN RESULTS FIGURE
# ============================================================

def plot_results(
    metrics: Dict,
    y_test: np.ndarray,
    out_dir: str
) -> None:
    """
    Genera la figura principale 2x2:

        A: anomaly score distribution
        B: confusion matrix
        C: ROC
        D: Precision-Recall
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

    # ========================================================
    # A — SCORE DISTRIBUTION
    # ========================================================

    ax = axes[0, 0]

    scores = metrics["scores"]

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

    ax.axvline(
        metrics["threshold"],
        linestyle="--",
        color="black",
        linewidth=2,
        label=(
            "IF Decision Threshold = "
            f"{metrics['threshold']:.3f}"
        )
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
    ax.grid(alpha=0.3)

    # ========================================================
    # B — CONFUSION MATRIX
    # ========================================================

    ax = axes[0, 1]

    cm = metrics["cm"]

    sns.heatmap(
        cm,
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
        "(B) Confusion Matrix",
        fontweight="bold"
    )

    # ========================================================
    # C — ROC
    # ========================================================

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
        linewidth=1.5,
        alpha=0.5
    )

    ax.fill_between(
        metrics["fpr"],
        metrics["tpr"],
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

    ax.grid(alpha=0.3)

    # ========================================================
    # D — PRECISION-RECALL
    # ========================================================

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
        linewidth=1.5,
        alpha=0.5,
        label=(
            f"Baseline = "
            f"{baseline:.3f}"
        )
    )

    ax.fill_between(
        metrics["recall"],
        metrics["precision"],
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

    ax.grid(alpha=0.3)

    # ========================================================
    # SAVE
    # ========================================================

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
# METRICS SUMMARY
# ============================================================

def plot_metrics_summary(
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Salva una tabella riassuntiva delle metriche
    slice-level.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.axis("tight")
    ax.axis("off")

    data = [

        [
            "AUROC",
            f"{metrics['auroc']:.4f}"
        ],

        [
            "Average Precision",
            f"{metrics['ap']:.4f}"
        ],

        [
            "F1 Score",
            f"{metrics['f1']:.4f}"
        ],

        [
            "Sensitivity (TPR)",
            f"{metrics['sensitivity']:.4f}"
        ],

        [
            "Specificity (TNR)",
            f"{metrics['specificity']:.4f}"
        ],

        [
            "Balanced Accuracy",
            f"{metrics['bacc']:.4f}"
        ],

        [
            "IF Decision Threshold",
            f"{metrics['threshold']:.4f}"
        ]
    ]

    table = ax.table(
        cellText=data,
        colLabels=[
            "Metric",
            "Value"
        ],
        cellLoc="center",
        loc="center",
        colWidths=[
            0.4,
            0.2
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
        2.5
    )

    for i in range(
        len(data) + 1
    ):

        if i == 0:

            table[
                (i, 0)
            ].set_facecolor(
                "#40466e"
            )

            table[
                (i, 1)
            ].set_facecolor(
                "#40466e"
            )

            table[
                (i, 0)
            ].set_text_props(
                weight="bold",
                color="white"
            )

            table[
                (i, 1)
            ].set_text_props(
                weight="bold",
                color="white"
            )

        elif i % 2 == 0:

            table[
                (i, 0)
            ].set_facecolor(
                "#f0f0f0"
            )

            table[
                (i, 1)
            ].set_facecolor(
                "#f0f0f0"
            )

    plt.title(
        "Isolation Forest Performance Summary",
        fontweight="bold",
        fontsize=12,
        pad=20
    )

    output_path = os.path.join(
        out_dir,
        "metrics_summary.png"
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
# SAVE SLICE-LEVEL ERROR ANALYSIS
# ============================================================

def save_error_analysis(
    y_test: np.ndarray,
    test_pids: np.ndarray,
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Salva le predizioni slice-level:

        TN
        FP
        FN
        TP

    insieme a:

        index
        true_label
        predicted_label
        anomaly_score
        decision_score

    Non modifica il modello.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    y_pred = metrics["y_pred"]
    scores = metrics["scores"]
    decision_scores = metrics["decision_scores"]

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
        "index": np.arange(len(y_test)),
        "patient_id": test_pids,  # <--- AGGIUNTO QUI
        "true_label": y_test,
        "predicted_label": y_pred,
        "category": categories,
        "anomaly_score": scores,
        "decision_score": decision_scores
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
# SAVE NUMERICAL RESULTS
# ============================================================

def save_metrics_csv(
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Salva le metriche slice-level in CSV.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

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

        "Balanced_Accuracy":
            metrics["bacc"],

        "Decision_Threshold":
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
    metrics: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    train_time: float
) -> None:
    """
    Salva report testuale dell'esperimento.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    report_path = os.path.join(
        out_dir,
        "isolation_forest_report.txt"
    )

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
            "No threshold tuning was performed.\n"
        )

        f.write(
            "The test set was used only for final evaluation.\n"
        )

        f.write(
            "The native Isolation Forest decision rule was used.\n"
        )

        f.write(
            "Decision threshold = 0.\n\n"
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
        # FEATURE ENGINEERING
        # ====================================================

        f.write(
            "FEATURE REPRESENTATION\n"
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
            "Input representation: flattened 64x64 grayscale image\n"
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
            f"Train samples: {X_train.shape[0]}\n"
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
            "Learning paradigm: unsupervised anomaly detection.\n"
        )

        f.write(
            "Training samples represent the normal class only.\n\n"
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
        # SLICE-LEVEL METRICS
        # ====================================================

        f.write(
            "SLICE-LEVEL METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"AUROC:                 "
            f"{metrics['auroc']:.4f}\n"
        )

        f.write(
            f"Average Precision:     "
            f"{metrics['ap']:.4f}\n"
        )

        f.write(
            f"F1 Score:              "
            f"{metrics['f1']:.4f}\n"
        )

        f.write(
            f"Sensitivity (Recall):  "
            f"{metrics['sensitivity']:.4f}\n"
        )

        f.write(
            f"Specificity:           "
            f"{metrics['specificity']:.4f}\n"
        )

        f.write(
            f"Balanced Accuracy:     "
            f"{metrics['bacc']:.4f}\n"
        )

        f.write(
            f"IF Decision Threshold: "
            f"{metrics['threshold']:.4f}\n\n"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        f.write(
            "SLICE-LEVEL CONFUSION MATRIX\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"True Negatives:  {metrics['tn']}\n"
        )

        f.write(
            f"False Positives: {metrics['fp']}\n"
        )

        f.write(
            f"False Negatives: {metrics['fn']}\n"
        )

        f.write(
            f"True Positives:  {metrics['tp']}\n\n"
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
            "The model was trained without access to tumor labels.\n"
        )

        f.write(
            "Tumor images were treated as potential anomalies during evaluation.\n"
        )

        f.write(
            "No test-set threshold optimization was performed.\n"
        )

        f.write(
            "Classification metrics correspond to the native\n"
        )

        f.write(
            "Isolation Forest decision rule.\n\n"
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

    X_train, y_train, train_pids = extract_features(
        train_ds
    )

    X_test, y_test, test_pids = extract_features(
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
    # VERIFY TRAINING LABELS
    # ========================================================

    if not np.all(
        y_train == 0
    ):

        raise ValueError(
            "ERROR: il training set contiene "
            "campioni tumorali. "
            "Per questa baseline il TRAIN "
            "deve contenere esclusivamente NORMAL."
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
        "  Decision threshold = 0"
    )

    print(
        "  No threshold tuning."
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
        "\n[5/5] Evaluation on independent test set..."
    )

    t0 = time.time()

    metrics = evaluate_iforest(
        model,
        X_test,
        y_test
    )

    eval_time = (
        time.time() - t0
    )

    # ========================================================
    # SLICE-LEVEL RESULTS
    # ========================================================

    print(
        "\n  SLICE-LEVEL RESULTS"
    )

    print(
        f"  AUROC: "
        f"{metrics['auroc']:.4f}"
    )

    print(
        f"  Average Precision: "
        f"{metrics['ap']:.4f}"
    )

    print(
        f"  F1: "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"  Sensitivity: "
        f"{metrics['sensitivity']:.4f}"
    )

    print(
        f"  Specificity: "
        f"{metrics['specificity']:.4f}"
    )

    print(
        f"  Balanced Accuracy: "
        f"{metrics['bacc']:.4f}"
    )

    print(
        f"  IF Decision Threshold: "
        f"{metrics['threshold']:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\n  CONFUSION MATRIX"
    )

    print(
        f"    TN: {metrics['tn']}"
    )

    print(
        f"    FP: {metrics['fp']}"
    )

    print(
        f"    FN: {metrics['fn']}"
    )

    print(
        f"    TP: {metrics['tp']}"
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

    plot_results(
        metrics,
        y_test,
        OUT_DIR
    )

    plot_metrics_summary(
        metrics,
        OUT_DIR
    )

    # ========================================================
    # ERROR ANALYSIS
    # ========================================================

    print(
        "\n[ERROR ANALYSIS] Saving slice-level predictions..."
    )

    save_error_analysis(
        y_test=y_test,
        test_pids=test_pids,
        metrics=metrics,
        out_dir=OUT_DIR
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    print(
        "\n[RESULTS] Saving numerical metrics..."
    )

    save_metrics_csv(
        metrics,
        OUT_DIR
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n[REPORTING] Saving report..."
    )

    save_report(
        metrics=metrics,
        X_train=X_train,
        y_train=y_train,
        y_test=y_test,
        out_dir=OUT_DIR,
        train_time=train_time
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
        "  • metrics_summary.png"
    )

    print(
        "  • isolation_forest_metrics.csv"
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
        "  • Threshold tuning: NO"
    )

    print(
        "  • Cross-validation: NO"
    )

    print(
        "  • Validation set: NO"
    )

    print(
        "  • Decision threshold: 0"
    )

    print(
        "  • Evaluation: SLICE-LEVEL ONLY"
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