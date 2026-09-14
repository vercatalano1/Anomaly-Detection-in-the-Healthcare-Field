# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS & PAIRWISE AUROC TEST
# ============================================================
#
# Analisi statistica finale e complementare:
#   1. Bootstrap IC95% per AUROC e Average Precision
#   2. Test di DeLong per confronti appaiati delle AUROC
#
# Confronti:
#   - PatchCore vs CNN Autoencoder
#   - PatchCore vs Isolation Forest
#
# ============================================================

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIGURAZIONE E PATH
# ============================================================

current_path = Path(__file__).resolve()

while current_path.name != "src" and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name == "src":
    PROJECT_ROOT = current_path.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "summary" / "bootstrap_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURAZIONE STATISTICA
# ============================================================

N_BOOTSTRAP = 2000
SEED = 42
ALPHA = 0.05  # IC 95%

REFERENCE_MODEL_NAME = "PatchCore"

SOURCES = [
    (
        "Isolation Forest",
        "ml_baseline/error_analysis.csv"
    ),
    (
        "CNN Autoencoder",
        "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"
    ),
    (
        "PatchCore",
        "patchcore/image_level_results.csv"
    ),
]

ORDER_MODELS = [
    "PatchCore",
    "CNN Autoencoder",
    "Isolation Forest"
]


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED
) -> Tuple[float, float, float]:

    rng = np.random.default_rng(seed)

    n = len(y_true)

    point_estimate = metric_fn(y_true, scores)

    bootstrap_values = []

    for _ in range(n_bootstrap):

        indices = rng.integers(0, n, size=n)

        y_boot = y_true[indices]
        scores_boot = scores[indices]

        # AUROC/AP non sono definiti se rimane una sola classe
        if len(np.unique(y_boot)) < 2:
            continue

        bootstrap_values.append(
            metric_fn(y_boot, scores_boot)
        )

    bootstrap_values = np.asarray(bootstrap_values)

    if len(bootstrap_values) == 0:
        raise RuntimeError(
            "Nessun bootstrap valido disponibile."
        )

    lower = np.percentile(
        bootstrap_values,
        100 * ALPHA / 2
    )

    upper = np.percentile(
        bootstrap_values,
        100 * (1 - ALPHA / 2)
    )

    return point_estimate, lower, upper


# ============================================================
# DELONG TEST
# ============================================================

def compute_midrank(x):
    """
    Calcola i midrank utilizzati dal test di DeLong.
    """
    x = np.asarray(x)

    order = np.argsort(x)
    sorted_x = x[order]

    midranks = np.empty(len(x), dtype=float)

    i = 0

    while i < len(sorted_x):

        j = i

        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1

        midranks[order[i:j]] = 0.5 * (i + j - 1) + 1

        i = j

    return midranks


def fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int
):
    """
    Implementazione del calcolo della varianza DeLong
    per ROC curve correlate.

    predictions_sorted_transposed:
        array [n_classifiers, n_samples]

    label_1_count:
        numero di campioni positivi.
    """

    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))

    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r] = compute_midrank(
            positive_predictions[r]
        )

        ty[r] = compute_midrank(
            negative_predictions[r]
        )

        tz[r] = compute_midrank(
            predictions_sorted_transposed[r]
        )

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (
        tz[:, :m] - tx
    ) / n

    v10 = 1.0 - (
        tz[:, m:] - ty
    ) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    if k == 1:
        sx = np.asarray([[sx]])
        sy = np.asarray([[sy]])

    covariance = sx / m + sy / n

    return aucs, covariance


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Test di DeLong per confrontare due AUROC correlate.

    Restituisce:
        auc_a
        auc_b
        p_value
    """

    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(y_true) != len(scores_a) or len(y_true) != len(scores_b):
        raise ValueError(
            "y_true e gli score devono avere la stessa lunghezza."
        )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Il test di DeLong richiede entrambe le classi."
        )

    # Ordiniamo i campioni mettendo prima i positivi
    order = np.argsort(-y_true)

    y_sorted = y_true[order]

    predictions = np.vstack([
        scores_a[order],
        scores_b[order]
    ])

    positive_count = int(np.sum(y_sorted))

    aucs, covariance = fast_delong(
        predictions,
        positive_count
    )

    auc_a = aucs[0]
    auc_b = aucs[1]

    variance_difference = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )

    if variance_difference <= 0:
        p_value = 1.0

    else:
        z = abs(auc_a - auc_b) / np.sqrt(
            variance_difference
        )

        # p-value bilaterale usando la normale standard
        from math import erf, sqrt

        p_value = 1.0 - erf(
            z / sqrt(2.0)
        )

    return auc_a, auc_b, p_value


# ============================================================
# PAIRED BOOTSTRAP PER DIFFERENZA AUROC
# ============================================================

def paired_bootstrap_delta(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_challenger: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED
) -> Tuple[float, float, float]:

    rng = np.random.default_rng(seed)

    n = len(y_true)

    delta_point = (
        roc_auc_score(y_true, scores_ref)
        -
        roc_auc_score(y_true, scores_challenger)
    )

    deltas = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_boot = y_true[indices]

        if len(np.unique(y_boot)) < 2:
            continue

        auc_ref = roc_auc_score(
            y_boot,
            scores_ref[indices]
        )

        auc_challenger = roc_auc_score(
            y_boot,
            scores_challenger[indices]
        )

        deltas.append(
            auc_ref - auc_challenger
        )

    deltas = np.asarray(deltas)

    if len(deltas) == 0:
        raise RuntimeError(
            "Nessun bootstrap valido per la differenza AUROC."
        )

    lower = np.percentile(
        deltas,
        100 * ALPHA / 2
    )

    upper = np.percentile(
        deltas,
        100 * (1 - ALPHA / 2)
    )

    return delta_point, lower, upper


# ============================================================
# FOREST PLOT
# ============================================================

def plot_forest(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    df_plot = (
        df
        .set_index("model")
        .reindex(ORDER_MODELS)
        .reset_index()
    )

    y_pos = np.arange(
        len(df_plot)
    )

    points = df_plot[metric].to_numpy()

    lower_errors = (
        points
        -
        df_plot[f"{metric}_ci_low"].to_numpy()
    )

    upper_errors = (
        df_plot[f"{metric}_ci_high"].to_numpy()
        -
        points
    )

    ax.errorbar(
        points,
        y_pos,
        xerr=[
            lower_errors,
            upper_errors
        ],
        fmt="o",
        markersize=9,
        capsize=5,
        capthick=2,
        elinewidth=2
    )

    ax.set_yticks(
        y_pos
    )

    ax.set_yticklabels(
        df_plot["model"],
        fontweight="bold"
    )

    ax.set_xlabel(
        "Score (95% CI)"
    )

    ax.set_title(
        title,
        fontweight="bold",
        pad=15
    )

    for i, row in df_plot.iterrows():

        value = row[metric]
        low = row[f"{metric}_ci_low"]
        high = row[f"{metric}_ci_high"]

        ax.text(
            high + 0.015,
            i,
            f"{value:.3f} [{low:.3f} - {high:.3f}]",
            va="center",
            fontsize=9
        )

    min_ci = df_plot[
        f"{metric}_ci_low"
    ].min()

    max_ci = df_plot[
        f"{metric}_ci_high"
    ].max()

    ax.set_xlim(
        max(0, min_ci - 0.1),
        min(1.0, max_ci + 0.25)
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.7
    )

    ax.grid(
        axis="y",
        visible=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PAIRWISE FOREST PLOT
# ============================================================

def plot_pairwise_forest(
    df_pair: pd.DataFrame
):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    y_pos = np.arange(
        len(df_pair)
    )

    points = df_pair[
        "Delta_AUROC"
    ].to_numpy()

    lower_errors = (
        points
        -
        df_pair["CI_Lower"].to_numpy()
    )

    upper_errors = (
        df_pair["CI_Upper"].to_numpy()
        -
        points
    )

    ax.errorbar(
        points,
        y_pos,
        xerr=[
            lower_errors,
            upper_errors
        ],
        fmt="s",
        markersize=8,
        capsize=5,
        capthick=2,
        elinewidth=2
    )

    # Linea dello zero:
    # nessuna differenza tra i due modelli.
    ax.axvline(
        0,
        linestyle="--",
        linewidth=2,
        alpha=0.7
    )

    ax.set_yticks(
        y_pos
    )

    ax.set_yticklabels(
        df_pair["Challenger"],
        fontweight="bold"
    )

    ax.set_xlabel(
        f"Δ AUROC ({REFERENCE_MODEL_NAME} − Challenger)"
    )

    ax.set_title(
        f"Confronto AUROC: {REFERENCE_MODEL_NAME} vs altri modelli",
        fontweight="bold",
        pad=15
    )

    for i, row in df_pair.iterrows():

        delta = row["Delta_AUROC"]
        low = row["CI_Lower"]
        high = row["CI_Upper"]
        p_value = row["p_value"]

        x_text = max(
            high,
            0
        ) + 0.015

        ax.text(
            x_text,
            i,
            f"Δ={delta:.3f}, p={p_value:.4f}",
            va="center",
            fontsize=9
        )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.7
    )

    ax.grid(
        axis="y",
        visible=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "forest_plot_pairwise.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "  ✓ Salvato: forest_plot_pairwise.png"
    )


# ============================================================
# ESECUZIONE PRINCIPALE
# ============================================================

def run():

    print("=" * 70)
    print(
        f"BOOTSTRAP CONFIDENCE INTERVALS "
        f"& DELONG TEST"
    )
    print(
        f"{N_BOOTSTRAP} resampling | IC {100 * (1 - ALPHA):.0f}%"
    )
    print("=" * 70)

    rows = []

    model_data = {}

    # --------------------------------------------------------
    # 1. CARICAMENTO DATI E BOOTSTRAP
    # --------------------------------------------------------

    for model_name, relative_path in SOURCES:

        csv_path = (
            RESULTS_DIR /
            relative_path
        )

        if not csv_path.exists():

            print(
                f"  [SKIP] {model_name}: "
                f"{csv_path} non trovato"
            )

            continue

        df = pd.read_csv(
            csv_path
        )

        required_columns = {
            "true_label",
            "anomaly_score"
        }

        missing = (
            required_columns
            -
            set(df.columns)
        )

        if missing:

            print(
                f"  [SKIP] {model_name}: "
                f"colonne mancanti {missing}"
            )

            continue

        y_true = df[
            "true_label"
        ].to_numpy()

        scores = df[
            "anomaly_score"
        ].to_numpy()

        if len(np.unique(y_true)) < 2:

            print(
                f"  [SKIP] {model_name}: "
                f"presente una sola classe"
            )

            continue

        model_data[
            model_name
        ] = df

        # AUROC
        auroc, auroc_lo, auroc_hi = (
            bootstrap_metric(
                y_true,
                scores,
                roc_auc_score
            )
        )

        # Average Precision
        ap, ap_lo, ap_hi = (
            bootstrap_metric(
                y_true,
                scores,
                average_precision_score
            )
        )

        rows.append({
            "model": model_name,

            "auroc": auroc,
            "auroc_ci_low": auroc_lo,
            "auroc_ci_high": auroc_hi,

            "ap": ap,
            "ap_ci_low": ap_lo,
            "ap_ci_high": ap_hi,
        })

        print(
            f"  [OK] {model_name:18s} | "
            f"AUROC = {auroc:.4f} "
            f"[{auroc_lo:.4f}, {auroc_hi:.4f}] | "
            f"AP = {ap:.4f} "
            f"[{ap_lo:.4f}, {ap_hi:.4f}]"
        )

    if not rows:
        print(
            "\nNessun risultato disponibile."
        )
        return

    result_df = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # SALVATAGGIO IC
    # --------------------------------------------------------

    result_df.to_csv(
        OUT_DIR /
        "bootstrap_confidence_intervals.csv",
        index=False
    )

    print(
        "\n✓ Salvato: "
        "bootstrap_confidence_intervals.csv"
    )

    # --------------------------------------------------------
    # FOREST PLOTS
    # --------------------------------------------------------

    print(
        "\nGenerazione Forest Plots..."
    )

    plot_forest(
        result_df,
        "auroc",
        "AUROC con Intervalli di Confidenza al 95%",
        "forest_plot_auroc.png"
    )

    plot_forest(
        result_df,
        "ap",
        "Average Precision con Intervalli di Confidenza al 95%",
        "forest_plot_ap.png"
    )

    print(
        "  ✓ Salvato: forest_plot_auroc.png"
    )

    print(
        "  ✓ Salvato: forest_plot_ap.png"
    )

    # --------------------------------------------------------
    # 2. CONFRONTI PAIRWISE
    # --------------------------------------------------------

    if REFERENCE_MODEL_NAME not in model_data:

        print(
            f"\n[WARNING] "
            f"{REFERENCE_MODEL_NAME} non disponibile."
        )

        return

    print("\n" + "=" * 70)
    print(
        f"PAIRWISE AUROC TEST: "
        f"{REFERENCE_MODEL_NAME} vs Altri"
    )
    print("=" * 70)

    df_ref = model_data[
        REFERENCE_MODEL_NAME
    ]

    y_ref = df_ref[
        "true_label"
    ].to_numpy()

    scores_ref = df_ref[
        "anomaly_score"
    ].to_numpy()

    pairwise_results = []

    for model_name, df_chal in model_data.items():

        if model_name == REFERENCE_MODEL_NAME:
            continue

        y_chal = df_chal[
            "true_label"
        ].to_numpy()

        scores_chal = df_chal[
            "anomaly_score"
        ].to_numpy()

        # ----------------------------------------------------
        # CONTROLLO FONDAMENTALE:
        # stesso numero di campioni e stesse label
        # ----------------------------------------------------

        if len(y_ref) != len(y_chal):

            print(
                f"  [SKIP] vs {model_name}: "
                f"numero di campioni differente"
            )

            continue

        if not np.array_equal(
            y_ref,
            y_chal
        ):

            print(
                f"  [SKIP] vs {model_name}: "
                f"le label non sono allineate"
            )

            continue

        # ----------------------------------------------------
        # DeLong
        # ----------------------------------------------------

        auc_ref, auc_chal, p_value = (
            delong_test(
                y_ref,
                scores_ref,
                scores_chal
            )
        )

        # ----------------------------------------------------
        # Bootstrap paired della differenza
        # ----------------------------------------------------

        delta, ci_low, ci_high = (
            paired_bootstrap_delta(
                y_ref,
                scores_ref,
                scores_chal,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED
            )
        )

        significant = (
            "SI"
            if p_value < ALPHA
            else "NO"
        )

        pairwise_results.append({

            "Reference": REFERENCE_MODEL_NAME,

            "Challenger": model_name,

            "AUROC_Reference": auc_ref,

            "AUROC_Challenger": auc_chal,

            "Delta_AUROC": delta,

            "CI_Lower": ci_low,

            "CI_Upper": ci_high,

            "p_value": p_value,

            "Significant": significant,
        })

        print(
            f"  [OK] vs {model_name:18s} | "
            f"Δ AUROC = {delta:.4f} | "
            f"95% CI = [{ci_low:.4f}, {ci_high:.4f}] | "
            f"p = {p_value:.4f} | "
            f"Sig = {significant}"
        )

    # --------------------------------------------------------
    # SALVATAGGIO RISULTATI PAIRWISE
    # --------------------------------------------------------

    if pairwise_results:

        df_pair = pd.DataFrame(
            pairwise_results
        )

        df_pair.to_csv(
            OUT_DIR /
            "pairwise_statistical_test.csv",
            index=False
        )

        print(
            "\n✓ Salvato: "
            "pairwise_statistical_test.csv"
        )

        plot_pairwise_forest(
            df_pair
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run()
