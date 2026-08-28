# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS (Full Models & Forest Plots)
# ============================================================
#
# Calcola intervalli di confidenza bootstrap (95%) per AUROC e AP.
# Genera grafici "Forest Plot", standard aureo nelle pubblicazioni
# biomediche per visualizzare l'incertezza statistica.
# ============================================================

import os
from typing import Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# CONFIGURAZIONE
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent

def find_project_root(current: Path) -> Path:
    for parent in [current] + list(current.parents):
        if (parent / "results").exists():
            return parent
    return current.parent

PROJECT_ROOT = find_project_root(CURRENT_DIR)
RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = os.path.join(RESULTS_DIR, "summary", "bootstrap_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP = 2000
SEED = 42
ALPHA = 0.05  # -> IC al 95%

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.facecolor"] = "white"

SOURCES = [
    ("Isolation Forest", "ml_baseline/error_analysis.csv"),
    ("CNN Autoencoder", "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"),
    ("CutPaste", "cutpaste/image_level_results.csv"),
    ("PatchCore", "patchcore/image_level_results.csv"),
]

ORDER_MODELS = ["PatchCore", "CutPaste", "CNN Autoencoder", "Isolation Forest"] # Invertito per asse Y dall'alto al basso

def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, scores)
    boot_values = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        y_boot = y_true[indices]
        s_boot = scores[indices]

        if len(np.unique(y_boot)) < 2:
            continue
        boot_values.append(metric_fn(y_boot, s_boot))

    boot_values = np.asarray(boot_values)
    lower = np.percentile(boot_values, 100 * ALPHA / 2)
    upper = np.percentile(boot_values, 100 * (1 - ALPHA / 2))

    return point_estimate, lower, upper

def load_scores(csv_path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(csv_path): return None
    df = pd.read_csv(csv_path)
    required = {"true_label", "anomaly_score"}
    if not required.issubset(df.columns): return None
    return df

def plot_forest(df: pd.DataFrame, metric: str, title: str, filename: str):
    plt.figure(figsize=(8, 5))
    
    # Prepariamo i dati
    df_plot = df.set_index("model").reindex(ORDER_MODELS).reset_index()
    
    y_pos = np.arange(len(df_plot))
    points = df_plot[metric]
    xerr_lower = points - df_plot[f"{metric}_ci_low"]
    xerr_upper = df_plot[f"{metric}_ci_high"] - points
    
    # Disegniamo il Forest Plot
    plt.errorbar(points, y_pos, xerr=[xerr_lower, xerr_upper], fmt='o', 
                 markersize=10, capsize=6, capthick=2, elinewidth=2, 
                 color='#2c3e50', markerfacecolor='#e74c3c', markeredgecolor='#c0392b')
    
    plt.yticks(y_pos, df_plot["model"], fontweight="bold")
    plt.xlabel("Score (95% CI)")
    plt.title(title, fontweight="bold", pad=15)
    
    # Aggiungiamo i valori numerici come etichette
    for i, (val, low, high) in enumerate(zip(points, df_plot[f"{metric}_ci_low"], df_plot[f"{metric}_ci_high"])):
        plt.text(high + 0.02, i, f"{val:.3f} [{low:.3f} - {high:.3f}]", va='center', fontsize=10)

    plt.xlim(max(0, df_plot[f"{metric}_ci_low"].min() - 0.1), min(1.0, df_plot[f"{metric}_ci_high"].max() + 0.25))
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', visible=False)
    plt.tight_layout()
    
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato grafico: {out_path}")

def run() -> Optional[pd.DataFrame]:
    print("=" * 70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} resample, IC 95%)")
    print("=" * 70)

    rows = []
    for model_name, relative_path in SOURCES:
        csv_path = os.path.join(RESULTS_DIR, relative_path)
        df = load_scores(csv_path)

        if df is None:
            print(f"  [SKIP] {model_name}: {csv_path} non trovato/incompatibile")
            continue

        y_true = df["true_label"].to_numpy()
        scores = df["anomaly_score"].to_numpy()

        auroc, auroc_lo, auroc_hi = bootstrap_metric(y_true, scores, roc_auc_score)
        ap, ap_lo, ap_hi = bootstrap_metric(y_true, scores, average_precision_score)

        rows.append({
            "model": model_name,
            "n_samples": len(df),
            "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
            "ap": ap, "ap_ci_low": ap_lo, "ap_ci_high": ap_hi,
        })
        print(f"  [OK]   {model_name:18s}: AUROC={auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

    if not rows:
        return None

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, "bootstrap_confidence_intervals.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\n✓ Salvato CSV: {out_path}")

    # Generiamo i Forest Plots
    print("\nGenerazione Forest Plots...")
    plot_forest(result_df, "auroc", "AUROC con Intervalli di Confidenza al 95%", "forest_plot_auroc.png")
    plot_forest(result_df, "ap", "Average Precision con Intervalli di Confidenza al 95%", "forest_plot_ap.png")

    return result_df

if __name__ == "__main__":
    run()