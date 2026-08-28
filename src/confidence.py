# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS & PAIRWISE TEST
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
# CONFIGURAZIONE E PATH ROBUSTI
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

ORDER_MODELS = ["PatchCore", "CutPaste", "CNN Autoencoder", "Isolation Forest"]
REFERENCE_MODEL_NAME = "PatchCore"  # Il "Campione" da testare contro gli altri

# ============================================================
# FUNZIONI BOOTSTRAP
# ============================================================
def bootstrap_metric(y_true: np.ndarray, scores: np.ndarray, metric_fn, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_estimate = metric_fn(y_true, scores)
    boot_values = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        y_boot = y_true[indices]
        s_boot = scores[indices]
        if len(np.unique(y_boot)) < 2: continue
        boot_values.append(metric_fn(y_boot, s_boot))

    boot_values = np.asarray(boot_values)
    lower = np.percentile(boot_values, 100 * ALPHA / 2)
    upper = np.percentile(boot_values, 100 * (1 - ALPHA / 2))
    return point_estimate, lower, upper

def pairwise_bootstrap_test(y_true: np.ndarray, scores_ref: np.ndarray, scores_chal: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> Tuple[float, float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    delta_point = roc_auc_score(y_true, scores_ref) - roc_auc_score(y_true, scores_chal)
    boot_deltas = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        y_boot = y_true[indices]
        if len(np.unique(y_boot)) < 2: continue
        auc_ref = roc_auc_score(y_boot, scores_ref[indices])
        auc_chal = roc_auc_score(y_boot, scores_chal[indices])
        boot_deltas.append(auc_ref - auc_chal)

    boot_deltas = np.asarray(boot_deltas)
    lower = np.percentile(boot_deltas, 100 * ALPHA / 2)
    upper = np.percentile(boot_deltas, 100 * (1 - ALPHA / 2))
    p_value = np.mean(boot_deltas <= 0) * 2 
    return delta_point, lower, upper, p_value

# ============================================================
# FUNZIONI DI PLOTTING
# ============================================================
def plot_forest(df: pd.DataFrame, metric: str, title: str, filename: str):
    plt.figure(figsize=(8, 5))
    df_plot = df.set_index("model").reindex(ORDER_MODELS).reset_index()
    y_pos = np.arange(len(df_plot))
    points = df_plot[metric]
    
    plt.errorbar(points, y_pos, xerr=[points - df_plot[f"{metric}_ci_low"], df_plot[f"{metric}_ci_high"] - points], 
                 fmt='o', markersize=10, capsize=6, capthick=2, elinewidth=2, 
                 color='#2c3e50', markerfacecolor='#e74c3c', markeredgecolor='#c0392b')
    
    plt.yticks(y_pos, df_plot["model"], fontweight="bold")
    plt.xlabel("Score (95% CI)")
    plt.title(title, fontweight="bold", pad=15)
    
    for i, (val, low, high) in enumerate(zip(points, df_plot[f"{metric}_ci_low"], df_plot[f"{metric}_ci_high"])):
        plt.text(high + 0.02, i, f"{val:.3f} [{low:.3f} - {high:.3f}]", va='center', fontsize=10)

    plt.xlim(max(0, df_plot[f"{metric}_ci_low"].min() - 0.1), min(1.0, df_plot[f"{metric}_ci_high"].max() + 0.25))
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', visible=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

def plot_pairwise_forest(df_pair: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(df_pair))
    points = df_pair["Delta_AUROC"]
    
    plt.errorbar(points, y_pos, xerr=[points - df_pair["CI_Lower"], df_pair["CI_Upper"] - points], 
                 fmt='s', markersize=8, capsize=6, capthick=2, elinewidth=2, 
                 color='#2c3e50', markerfacecolor='#3498db', markeredgecolor='#2980b9')
    
    # La fatidica linea dello ZERO (Nessuna differenza)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label="Ipotesi Nulla (Nessuna Differenza)")
    
    plt.yticks(y_pos, df_pair["Challenger"], fontweight="bold")
    plt.xlabel(f"Vantaggio (Δ AUROC) di {REFERENCE_MODEL_NAME} (95% CI)")
    plt.title(f"Significatività Statistica vs {REFERENCE_MODEL_NAME}", fontweight="bold", pad=15)
    
    for i, (val, low, high, p) in enumerate(zip(points, df_pair["CI_Lower"], df_pair["CI_Upper"], df_pair["p_value"])):
        plt.text(max(high, 0) + 0.02, i, f"Δ={val:.3f} (p={p:.4f})", va='center', fontsize=10)

    plt.legend(loc="lower right")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.grid(axis='y', visible=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "forest_plot_pairwise.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato grafico pairwise: forest_plot_pairwise.png")

# ============================================================
# ESECUZIONE
# ============================================================
def run():
    print("=" * 70)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS & PAIRWISE TEST ({N_BOOTSTRAP} resample, IC 95%)")
    print("=" * 70)

    rows = []
    model_data = {}

    # 1. Intervalli Singoli
    for model_name, relative_path in SOURCES:
        csv_path = RESULTS_DIR / relative_path
        if not csv_path.exists():
            print(f"  [SKIP] {model_name}: {csv_path} non trovato")
            continue
            
        df = pd.read_csv(csv_path)
        model_data[model_name] = df
        y_true = df["true_label"].to_numpy()
        scores = df["anomaly_score"].to_numpy()

        auroc, auroc_lo, auroc_hi = bootstrap_metric(y_true, scores, roc_auc_score)
        ap, ap_lo, ap_hi = bootstrap_metric(y_true, scores, average_precision_score)

        rows.append({
            "model": model_name,
            "auroc": auroc, "auroc_ci_low": auroc_lo, "auroc_ci_high": auroc_hi,
            "ap": ap, "ap_ci_low": ap_lo, "ap_ci_high": ap_hi,
        })
        print(f"  [OK]   {model_name:18s}: AUROC={auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]")

    if not rows: return
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "bootstrap_confidence_intervals.csv", index=False)
    
    print("\nGenerazione Forest Plots...")
    plot_forest(result_df, "auroc", "AUROC con Intervalli di Confidenza al 95%", "forest_plot_auroc.png")
    plot_forest(result_df, "ap", "Average Precision con Intervalli di Confidenza al 95%", "forest_plot_ap.png")

    # 2. Test Statistico Pairwise
    if REFERENCE_MODEL_NAME in model_data:
        print("\n" + "=" * 70)
        print(f"PAIRWISE TEST: {REFERENCE_MODEL_NAME} vs Altri")
        print("=" * 70)
        
        df_ref = model_data[REFERENCE_MODEL_NAME]
        pairwise_results = []
        
        for model_name, df_chal in model_data.items():
            if model_name == REFERENCE_MODEL_NAME: continue
            
            delta, lo, hi, p_val = pairwise_bootstrap_test(
                df_ref["true_label"].to_numpy(), 
                df_ref["anomaly_score"].to_numpy(), 
                df_chal["anomaly_score"].to_numpy()
            )
            significativo = "SI" if p_val < ALPHA else "NO"
            pairwise_results.append({
                "Challenger": model_name, "Delta_AUROC": delta,
                "CI_Lower": lo, "CI_Upper": hi, "p_value": p_val, "Significant": significativo
            })
            print(f"  [OK] vs {model_name:18s} | Δ AUROC = {delta:.3f} | p-val: {p_val:.4f} | Sig: {significativo}")
        
        if pairwise_results:
            df_pair = pd.DataFrame(pairwise_results)
            df_pair.to_csv(OUT_DIR / "pairwise_statistical_test.csv", index=False)
            plot_pairwise_forest(df_pair)

if __name__ == "__main__":
    run()