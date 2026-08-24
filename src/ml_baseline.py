# ============================================================
# ISOLATION FOREST — PURE UNSUPERVISED BASELINE
# ============================================================
#
# Baseline metodologica:
#   - TRAIN: esclusivamente immagini NORMAL
#   - TEST: NORMAL + TUMOR
#   - Isolation Forest standard
#   - contamination="auto"
#   - nessuna cross-validation
#   - nessun validation set
#   - nessun threshold tuning
#   - soglia decisionale naturale di Isolation Forest = 0
#
# Obiettivo:
# valutare quanto un modello addestrato esclusivamente
# su immagini normali riesca a identificare le immagini
# tumorali come anomalie.
#
# ============================================================


import os
import time
from typing import Dict, Tuple

import numpy as np
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

from dataloader import get_dataset


# ==========================================================
# CONFIGURAZIONE
# ==========================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

sns.set_theme(
    style="whitegrid",
    font_scale=1.0
)

plt.rcParams["figure.facecolor"] = "white"


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_features(
    dataset,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estrae le immagini e le relative label.

    Input:
        dataset: BraTSDataset

    Output:
        X: feature [N, D]
        y: label [N]

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

    for batch in loader:

        imgs = batch["img"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        # [B, 1, 64, 64] -> [B, 4096]
        X_list.append(
            imgs.reshape(
                imgs.shape[0],
                -1
            )
        )

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


# ==========================================================
# NORMALIZATION
# ==========================================================

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
    Standardizzazione delle feature.

    IMPORTANTE:
    lo scaler viene FITTATO esclusivamente sul TRAIN.

    Il TEST viene solamente trasformato.

    Questo evita leakage.
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


# ==========================================================
# ISOLATION FOREST
# ==========================================================

def train_iforest(
    X_train: np.ndarray
) -> IsolationForest:
    """
    Addestra Isolation Forest.

    contamination="auto":
        utilizza la configurazione standard
        dell'algoritmo.

    Nessuna informazione sulle label tumorali
    viene utilizzata durante il training.
    """

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )

    model.fit(
        X_train
    )

    return model


# ==========================================================
# EVALUATION
# ==========================================================
def evaluate_iforest(
    model: IsolationForest,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Valuta Isolation Forest sul test set.

    Metodologia:
        - decision_function > 0  -> normal
        - decision_function < 0  -> anomaly
        - predict() di sklearn   -> stessa regola

    Per AUROC/AP il segno viene invertito:
        score alto -> più anomalo

    Nessun threshold tuning.
    """

    # ======================================================
    # ISOLATION FOREST DECISION FUNCTION
    # ======================================================

    decision_scores = model.decision_function(X_test)

    # decision_function:
    #   valore alto  -> più normale
    #   valore basso -> più anomalo
    #
    # Per le metriche anomaly detection invertiamo il segno:
    #   score alto -> più anomalo

    scores = -decision_scores

    # Soglia naturale di Isolation Forest.
    #
    # Sul decision_function:
    #   > 0 = normal
    #   < 0 = anomaly
    #
    # Sullo score invertito:
    #   < 0 = normal
    #   > 0 = anomaly

    threshold = 0.0

    # ======================================================
    # PREDICTION
    # ======================================================

    # Usiamo direttamente la predizione nativa di sklearn.
    #
    # sklearn:
    #   +1 = normal
    #   -1 = anomaly

    y_pred_raw = model.predict(X_test)

    # Conversione:
    #   normal  -> 0
    #   anomaly -> 1

    y_pred = (y_pred_raw == -1).astype(int)

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    # ======================================================
    # AUROC
    # ======================================================

    auroc = roc_auc_score(
        y_test,
        scores
    )

    # ======================================================
    # AVERAGE PRECISION
    # ======================================================

    ap = average_precision_score(
        y_test,
        scores
    )

    # ======================================================
    # SENSITIVITY
    # ======================================================

    sensitivity = (
        tp /
        (tp + fn + 1e-8)
    )

    # ======================================================
    # SPECIFICITY
    # ======================================================

    specificity = (
        tn /
        (tn + fp + 1e-8)
    )

    # ======================================================
    # PRECISION
    # ======================================================

    precision_value = (
        tp /
        (tp + fp + 1e-8)
    )

    # ======================================================
    # F1
    # ======================================================

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

    # ======================================================
    # BALANCED ACCURACY
    # ======================================================

    bacc = balanced_accuracy_score(
        y_test,
        y_pred
    )

    # ======================================================
    # ROC
    # ======================================================

    fpr, tpr, roc_thresholds = roc_curve(
        y_test,
        scores
    )

    # ======================================================
    # PRECISION-RECALL
    # ======================================================

    precision_curve, recall_curve, pr_thresholds = (
        precision_recall_curve(
            y_test,
            scores
        )
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

        # Natural IF threshold
        "threshold": threshold,

        # Scores
        "scores": scores,
        "decision_scores": decision_scores,

        # Predictions
        "y_pred": y_pred,

        # Confusion matrix
        "cm": cm,

        # ROC
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,

        # PR
        "precision": precision_curve,
        "recall": recall_curve,
        "pr_thresholds": pr_thresholds,

        # Confusion matrix values
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


# ==========================================================
# MAIN RESULTS FIGURE
# ==========================================================

def plot_results(
    metrics: Dict,
    y_test: np.ndarray,
    out_dir: str
) -> None:
    """
    Genera la figura principale 2x2.

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

    # ======================================================
    # A — SCORE DISTRIBUTION
    # ======================================================

    ax = axes[0, 0]

    scores = metrics["scores"]

    sns.kdeplot(
        scores[y_test == 0],
        fill=True,
        color="#2ca02c",
        alpha=0.4,
        label="Healthy",
        ax=ax,
        linewidth=2
    )

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
            f"IF Decision Threshold = "
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

    ax.grid(
        alpha=0.3
    )

    # ======================================================
    # B — CONFUSION MATRIX
    # ======================================================

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

    # ======================================================
    # C — ROC
    # ======================================================

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

    ax.grid(
        alpha=0.3
    )

    # ======================================================
    # D — PRECISION RECALL
    # ======================================================

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


# ==========================================================
# METRICS SUMMARY
# ==========================================================

def plot_metrics_summary(
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Tabella riassuntiva delle metriche.
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

        else:

            if i % 2 == 0:

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


# ==========================================================
# QUALITATIVE ERROR ANALYSIS
# ==========================================================

def save_error_analysis(
    test_ds,
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Salva gli indici dei:

        TN
        FP
        FN
        TP

    per una successiva analisi qualitativa.

    Non modifica il modello.
    """

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    y_pred = metrics["y_pred"]

    # ------------------------------------------------------
    # INDICI
    # ------------------------------------------------------

    tn_idx = np.where(
        (metrics["cm"] is not None)
    )[0] if False else None

    # Ricaviamo direttamente le categorie
    # usando le label e le predizioni.

    # ATTENZIONE:
    # test_ds deve avere lo stesso ordine
    # utilizzato durante extract_features.

    # Per evitare ambiguità, salviamo semplicemente
    # gli indici dei campioni.

    # Gli indici reali vengono recuperati in main.

    return


# ==========================================================
# REPORT
# ==========================================================

def save_report(
    metrics: Dict,
    X_train: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
    train_time: float
) -> None:
    """
    Salva report testuale completo.
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

        # --------------------------------------------------
        # METHODOLOGY
        # --------------------------------------------------

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
            "No validation set was used for threshold tuning.\n"
        )

        f.write(
            "The test set was used only for final evaluation.\n"
        )

        f.write(
            "The natural Isolation Forest decision threshold was used.\n"
        )

        f.write(
            "Decision threshold = 0.\n\n"
        )

        # --------------------------------------------------
        # HYPERPARAMETERS
        # --------------------------------------------------

        f.write(
            "HYPERPARAMETERS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "n_estimators: 200\n"
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

        # --------------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------------

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
            "Input image representation: flattened 64x64 image\n"
        )

        f.write(
            "Normalization: StandardScaler\n"
        )

        f.write(
            "Scaler fitted exclusively on training data.\n\n"
        )

        # --------------------------------------------------
        # TRAINING
        # --------------------------------------------------

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
            f"Training time: {train_time:.2f}s\n"
        )

        f.write(
            "Learning paradigm: one-class / unsupervised anomaly detection.\n"
        )

        f.write(
            "Training samples represent the normal class only.\n\n"
        )

        # --------------------------------------------------
        # TEST
        # --------------------------------------------------

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

        # --------------------------------------------------
        # METRICS
        # --------------------------------------------------

        f.write(
            "METRICS\n"
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

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        f.write(
            "CONFUSION MATRIX\n"
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

        # --------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------

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
            "The decision threshold was not optimized using the test set.\n"
        )

        f.write(
            "The reported classification metrics therefore correspond\n"
        )

        f.write(
            "to the native Isolation Forest decision rule.\n\n"
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


# ==========================================================
# SAVE NUMERICAL RESULTS
# ==========================================================

def save_metrics_csv(
    metrics: Dict,
    out_dir: str
) -> None:
    """
    Salva le metriche in formato CSV.
    """

    import pandas as pd

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


# ==========================================================
# MAIN EXPERIMENT
# ==========================================================

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

    out_dir = (
        "results/isolation_forest"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    # ======================================================
    # 1. DATA LOADING
    # ======================================================

    print(
        "\n[1/5] Loading datasets..."
    )

    t0 = time.time()

    train_ds = get_dataset(
        "brats",
        data_root="data",
        mode="train"
    )

    test_ds = get_dataset(
        "brats",
        data_root="data",
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

    # ======================================================
    # 2. FEATURE EXTRACTION
    # ======================================================

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

    print(
        f"  ✓ X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"  ✓ X_test shape:  "
        f"{X_test.shape}"
    )

    print(
        f"  ✓ Train labels:"
        f" {np.unique(y_train, return_counts=True)}"
    )

    print(
        f"  ✓ Test labels:"
        f" {np.unique(y_test, return_counts=True)}"
    )

    print(
        f"  Feature extraction time: "
        f"{time.time() - t0:.2f}s"
    )

    # ======================================================
    # 3. NORMALIZATION
    # ======================================================

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

    # ======================================================
    # 4. TRAINING
    # ======================================================

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

    # ======================================================
    # 5. EVALUATION
    # ======================================================

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

    # ------------------------------------------------------
    # PRINT METRICS
    # ------------------------------------------------------

    print(
        f"\n  AUROC: "
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

    print(
        f"  Evaluation time: "
        f"{eval_time:.2f}s"
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

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

    # ======================================================
    # VISUALIZATION
    # ======================================================

    print(
        "\n[VISUALIZATION] Generating plots..."
    )

    plot_results(
        metrics,
        y_test,
        out_dir
    )

    plot_metrics_summary(
        metrics,
        out_dir
    )

    # ======================================================
    # SAVE METRICS
    # ======================================================

    print(
        "\n[RESULTS] Saving numerical metrics..."
    )

    save_metrics_csv(
        metrics,
        out_dir
    )

    # ======================================================
    # REPORT
    # ======================================================

    print(
        "\n[REPORTING] Saving report..."
    )

    save_report(
        metrics,
        X_train,
        y_test,
        out_dir,
        train_time
    )

    # ======================================================
    # SUMMARY
    # ======================================================

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
        f"{out_dir}/"
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
        "  • Decision threshold: 0"
    )

    print(
        "\n" +
        "=" * 70 +
        "\n"
    )


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    run_experiment()