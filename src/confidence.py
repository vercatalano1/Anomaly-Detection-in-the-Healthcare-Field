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
# CONFIGURAZIONE STATISTICA E GRAFICA
# ============================================================

N_BOOTSTRAP = 2000
SEED = 42
ALPHA = 0.05  # IC 95%
REFERENCE_MODEL_NAME = "PatchCore"

SOURCES = [
    ("Isolation Forest", "ml_baseline/error_analysis.csv"),
    ("CNN Autoencoder", "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"),
    ("PatchCore", "patchcore/image_level_results.csv"),
]

ORDER_MODELS = ["PatchCore", "CNN Autoencoder", "Isolation Forest"]

# Aggiunta della palette unificata (coerente con patient_level.py)
MODEL_COLORS = {
    "Isolation Forest": "#d62728",  # Rosso morbido
    "CNN Autoencoder": "#2ca02c",   # Verde
    "PatchCore": "#1f77b4"          # Blu
}

# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_metric(y_true, scores, metric_fn, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, scores)
    bootstrap_values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_boot = y_true[indices]
        scores_boot = scores[indices]
        if len(np.unique(y_boot)) < 2:
            continue
        bootstrap_values.append(metric_fn(y_boot, scores_boot))

    bootstrap_values = np.asarray(bootstrap_values)
    if len(bootstrap_values) == 0:
        raise RuntimeError("Nessun bootstrap valido disponibile.")

    lower = np.percentile(bootstrap_values, 100 * ALPHA / 2)
    upper = np.percentile(bootstrap_values, 100 * (1 - ALPHA / 2))
    return point_estimate, lower, upper


# ============================================================
# DELONG TEST (FAST IMPLEMENTATION)
# ============================================================

def compute_midrank(x):
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

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r] = compute_midrank(positive_predictions[r])
        ty[r] = compute_midrank(negative_predictions[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.asarray([[sx]])
        sy = np.asarray([[sy]])
    covariance = sx / m + sy / n
    return aucs, covariance

def delong_test(y_true, scores_a, scores_b):
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    predictions = np.vstack([scores_a[order], scores_b[order]])
    positive_count = int(np.sum(y_sorted))

    aucs, covariance = fast_delong(predictions, positive_count)
    auc_a = aucs[0]
    auc_b = aucs[1]
    variance_difference = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]

    if variance_difference <= 0:
        p_value = 1.0
    else:
        z = abs(auc_a - auc_b) / np.sqrt(variance_difference)
        from math import erf, sqrt
        p_value = 1.0 - erf(z / sqrt(2.0))

    return auc_a, auc_b, p_value

# ============================================================
# PAIRED BOOTSTRAP PER DIFFERENZA AUROC
# ============================================================

def paired_bootstrap_delta(y_true, scores_ref, scores_challenger, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    delta_point = roc_auc_score(y_true, scores_ref) - roc_auc_score(y_true, scores_challenger)
    deltas = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_boot = y_true[indices]
        if len(np.unique(y_boot)) < 2:
            continue
        auc_ref = roc_auc_score(y_boot, scores_ref[indices])
        auc_challenger = roc_auc_score(y_boot, scores_challenger[indices])
        deltas.append(auc_ref - auc_challenger)

    deltas = np.asarray(deltas)
    lower = np.percentile(deltas, 100 * ALPHA / 2)
    upper = np.percentile(deltas, 100 * (1 - ALPHA / 2))
    return delta_point, lower, upper


# ============================================================
# MIGLIORAMENTO GRAFICO: FOREST PLOT (STILE ACCADEMICO AVANZATO)
# ============================================================

def plot_forest(df: pd.DataFrame, metric: str, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    df_plot = df.set_index("model").reindex(ORDER_MODELS).reset_index()
    y_pos = np.arange(len(df_plot))

    # Sfondo a strisce morbide per facilitare la lettura
    for i in range(len(df_plot)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f8f9fa', zorder=0)

    points = df_plot[metric].to_numpy()
    lower_errors = points - df_plot[f"{metric}_ci_low"].to_numpy()
    upper_errors = df_plot[f"{metric}_ci_high"].to_numpy() - points

    # --- NOVITÀ: Linea verticale di riferimento sul modello migliore ---
    if REFERENCE_MODEL_NAME in df_plot["model"].values:
        best_val = df_plot.loc[df_plot["model"] == REFERENCE_MODEL_NAME, metric].values[0]
        ax.axvline(best_val, linestyle=":", color=MODEL_COLORS.get(REFERENCE_MODEL_NAME, "gray"), linewidth=2, zorder=1, alpha=0.6)

    # Disegna ogni punto con il suo colore specifico
    for i, model in enumerate(df_plot["model"]):
        color = MODEL_COLORS.get(model, "#333333")
        ax.errorbar(
            points[i], i, 
            xerr=[[lower_errors[i]], [upper_errors[i]]],
            fmt="D", markersize=9, capsize=6, capthick=2.5, elinewidth=2.5, 
            color=color, zorder=3, markeredgecolor='white', markeredgewidth=1
        )

        value = points[i]
        low = df_plot.loc[i, f"{metric}_ci_low"]
        high = df_plot.loc[i, f"{metric}_ci_high"]

        # Testo del CI a destra
        ax.text(
            max(points) + max(upper_errors) + 0.02, i, 
            f"{value:.3f} [{low:.3f}, {high:.3f}]", 
            va="center", fontsize=10, fontweight='normal', color='#333333'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["model"], fontweight="bold", fontsize=11)
    
    # Colora le etichette dell'asse Y per richiamare i modelli
    for label in ax.get_yticklabels():
        label.set_color(MODEL_COLORS.get(label.get_text(), "#333333"))

    ax.set_xlabel(f"{metric.upper()} con 95% CI", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=15, fontsize=13)
    
    # Assicura che ci sia spazio per il testo a destra
    min_ci = df_plot[f"{metric}_ci_low"].min()
    max_ci = df_plot[f"{metric}_ci_high"].max()
    ax.set_xlim(max(0, min_ci - 0.05), min(1.0, max_ci + 0.15))

    ax.grid(axis="x", linestyle="--", alpha=0.5, color='gray', zorder=1)
    ax.grid(axis="y", visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# MIGLIORAMENTO GRAFICO: PAIRWISE FOREST PLOT (STILE ACCADEMICO)
# ============================================================

def plot_pairwise_forest(df_pair: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = np.arange(len(df_pair))
    
    points = df_pair["Delta_AUROC"].to_numpy()
    lower_errors = points - df_pair["CI_Lower"].to_numpy()
    upper_errors = df_pair["CI_Upper"].to_numpy() - points

    # Sfondo a strisce morbide
    for i in range(len(df_pair)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f8f9fa', zorder=0)

    # Linea dello zero molto evidente ma neutra (Nessuna differenza)
    ax.axvline(0, linestyle="--", color="#333333", linewidth=1.5, alpha=0.8, zorder=1)

    for i, model in enumerate(df_pair["Challenger"]):
        color = MODEL_COLORS.get(model, "#333333")
        ax.errorbar(
            points[i], i, 
            xerr=[[lower_errors[i]], [upper_errors[i]]],
            fmt="s", markersize=9, capsize=6, capthick=2.5, elinewidth=2.5, 
            color=color, zorder=3, markeredgecolor='white', markeredgewidth=1
        )

        delta = points[i]
        low = df_pair.loc[i, "CI_Lower"]
        high = df_pair.loc[i, "CI_Upper"]
        p_value = df_pair.loc[i, "p_value"]

        # Formattazione accademica per la significatività (asterischi)
        if p_value < 0.001:
            sig_stars = "***"
            p_text = "< 0.001"
        elif p_value < 0.01:
            sig_stars = "**"
            p_text = f"= {p_value:.3f}"
        elif p_value < 0.05:
            sig_stars = "*"
            p_text = f"= {p_value:.3f}"
        else:
            sig_stars = "(ns)"
            p_text = f"= {p_value:.3f}"

        # Testo tutto in nero/grigio scuro come nei paper
        ax.text(
            max(high, 0) + 0.02, i, 
            f"Δ={delta:.3f} [{low:.3f}, {high:.3f}] | p {p_text} {sig_stars}", 
            va="center", fontsize=10, color='#333333'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pair["Challenger"], fontweight="bold", fontsize=11)
    
    # Colora solo il nome del modello sull'asse per richiamare la legenda
    for label in ax.get_yticklabels():
        label.set_color(MODEL_COLORS.get(label.get_text(), "#333333"))

    ax.set_xlabel(f"Δ AUROC ({REFERENCE_MODEL_NAME} − Modello Challenger)", fontweight="bold", fontsize=11)
    ax.set_title(f"Test di DeLong: {REFERENCE_MODEL_NAME} vs Altri Modelli", fontweight="bold", pad=15, fontsize=13)
    
    ax.grid(axis="x", linestyle="--", alpha=0.5, color='gray', zorder=1)
    ax.grid(axis="y", visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "forest_plot_pairwise.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Salvato: forest_plot_pairwise.png")


# ============================================================
# ESECUZIONE PRINCIPALE
# ============================================================

def run():
    print("=" * 70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS & DELONG TEST")
    print(f"{N_BOOTSTRAP} resampling | IC {100 * (1 - ALPHA):.0f}%")
    print("=" * 70)

    rows = []
    model_data = {}

    for model_name, relative_path in SOURCES:
        csv_path = RESULTS_DIR / relative_path
        if not csv_path.exists():
            print(f"  [SKIP] {model_name}: {csv_path} non trovato")
            continue

        df = pd.read_csv(csv_path)
        required_columns = {"true_label", "anomaly_score"}
        if required_columns - set(df.columns):
            continue

        y_true = df["true_label"].to_numpy()
        scores = df["anomaly_score"].to_numpy()

        if len(np.unique(y_true)) < 2:
            continue
        model_data[model_name] = df

        auroc, auroc_lo, auroc_hi = bootstrap_metric(y_true, scores, roc_auc_score)
        ap, ap_lo, ap_hi = bootstrap_metric(y_true, scores, average_precision_score)

        rows.append({
            "model": model_name,
            "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
            "ap": ap, "ap_ci_low": ap_lo, "ap_ci_high": ap_hi,
        })
        print(f"  [OK] {model_name:18s} | AUROC = {auroc:.4f} | AP = {ap:.4f}")

    if not rows:
        return

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("\n✓ Salvato: bootstrap_confidence_intervals.csv")

    print("\nGenerazione Forest Plots (Enhanced Style)...")
    plot_forest(result_df, "auroc", "AUROC con Intervalli di Confidenza al 95%", "forest_plot_auroc.png")
    plot_forest(result_df, "ap", "Average Precision con Intervalli di Confidenza al 95%", "forest_plot_ap.png")
    print("  ✓ Salvato: forest_plot_auroc.png e forest_plot_ap.png")

    if REFERENCE_MODEL_NAME not in model_data:
        return

    print("\n" + "=" * 70)
    print(f"PAIRWISE AUROC TEST: {REFERENCE_MODEL_NAME} vs Altri")
    print("=" * 70)

    df_ref = model_data[REFERENCE_MODEL_NAME]
    y_ref = df_ref["true_label"].to_numpy()
    scores_ref = df_ref["anomaly_score"].to_numpy()
    pairwise_results = []

    for model_name, df_chal in model_data.items():
        if model_name == REFERENCE_MODEL_NAME:
            continue
        y_chal = df_chal["true_label"].to_numpy()
        scores_chal = df_chal["anomaly_score"].to_numpy()

        if len(y_ref) != len(y_chal) or not np.array_equal(y_ref, y_chal):
            continue

        auc_ref, auc_chal, p_value = delong_test(y_ref, scores_ref, scores_chal)
        delta, ci_low, ci_high = paired_bootstrap_delta(y_ref, scores_ref, scores_chal)
        significant = "SI" if p_value < ALPHA else "NO"

        pairwise_results.append({
            "Reference": REFERENCE_MODEL_NAME, "Challenger": model_name,
            "AUROC_Reference": auc_ref, "AUROC_Challenger": auc_chal,
            "Delta_AUROC": delta, "CI_Lower": ci_low, "CI_Upper": ci_high,
            "p_value": p_value, "Significant": significant,
        })
        print(f"  [OK] vs {model_name:18s} | Δ AUROC = {delta:.4f} | p = {p_value:.4f}")

    if pairwise_results:
        df_pair = pd.DataFrame(pairwise_results)
        df_pair.to_csv(OUT_DIR / "pairwise_statistical_test.csv", index=False)
        plot_pairwise_forest(df_pair)

if __name__ == "__main__":
    run()