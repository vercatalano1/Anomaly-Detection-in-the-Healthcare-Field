# ============================================================
# PATIENT-LEVEL ANALYSIS (Full Models Comparison & Advanced Plots)
# ============================================================
#
# Aggrega i risultati slice-level a livello di paziente e ricalcola
# AUROC/AP, confrontandoli con i valori slice-level per tutti i modelli.
# Genera metriche CSV e grafici di livello accademico per la tesi.
#
# Regola di aggregazione:
#   - patient_score = max(anomaly_score) sulle slice del paziente
#   - patient_label = 1 se ALMENO una slice del paziente e' tumorale
# ============================================================

import os
from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ============================================================
# CONFIGURAZIONE E PATH
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent

def find_project_root(current: Path) -> Path:
    for parent in [current] + list(current.parents):
        if (parent / "results").exists():
            return parent
    return current.parent

PROJECT_ROOT = find_project_root(CURRENT_DIR)
RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = os.path.join(RESULTS_DIR, "summary", "patient_level")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 11})

# ============================================================
# SORGENTI E PALETTE UNIFICATA
# ============================================================

COMPATIBLE_SOURCES = [
    ("Isolation Forest", "ml_baseline/error_analysis.csv"),
    ("CNN Autoencoder", "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"),
    ("PatchCore", "patchcore/image_level_results.csv"),
]

ORDER_MODELS = ["Isolation Forest", "CNN Autoencoder", "PatchCore"]

# Palette ufficiale coerente con tutta la tesi
MODEL_COLORS = {
    "Isolation Forest": "#d62728",
    "CNN Autoencoder": "#2ca02c",
    "PatchCore": "#1f77b4"
}


def compute_patient_level(df: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_id", "true_label", "anomaly_score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Colonne mancanti nel CSV: {missing}")

    grouped = df.groupby("patient_id").agg(
        patient_label=("true_label", "max"),
        patient_score=("anomaly_score", "max"),
        n_slices=("true_label", "count"),
    ).reset_index()

    return grouped


def evaluate(df: pd.DataFrame) -> dict:
    return {
        "auroc": roc_auc_score(df["true_label"],
                                df["anomaly_score"] 
        ),
        "ap": average_precision_score(df["true_label"] ,
                                      df["anomaly_score"]
        ),
    }


def plot_comparison_bar(df, metric_slice, metric_patient, title, filename, ylabel="Score"):
    """
    Funzione per generare barplot raggruppati accademici.
    Slice-level = Trasparente/Chiaro, Patient-level = Colore Pieno/Scuro
    """
    models = [m for m in ORDER_MODELS if m in df["model"].values]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    slice_vals = [df[df["model"]==m][metric_slice].values[0] for m in models]
    patient_vals = [df[df["model"]==m][metric_patient].values[0] for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    
    # Barre Slice-level (più chiare)
    rects1 = ax.bar(x - width/2, slice_vals, width, label='Slice-level', 
                    color=colors, alpha=0.4, edgecolor=colors, linewidth=2)
    
    # Barre Patient-level (colore pieno)
    rects2 = ax.bar(x + width/2, patient_vals, width, label='Patient-level', 
                    color=colors, edgecolor='white', linewidth=1)
    
    # Etichette valori
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, color='#333333')

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight="bold", pad=15, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(0, 1.20)
    
    # Legenda personalizzata neutrale
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#333333', alpha=0.4, edgecolor='#333333', linewidth=2, label='Slice-level (Singola Immagine)'),
        Patch(facecolor='#333333', edgecolor='white', linewidth=1, label='Patient-level (Max Pooling)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False)
    
    ax.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()


def run() -> Optional[pd.DataFrame]:
    print("=" * 70)
    print("PATIENT-LEVEL ANALYSIS — Tutti i modelli")
    print("=" * 70)

    rows = []
    patient_data_dict = {}

    for model_name, relative_path in COMPATIBLE_SOURCES:
        csv_path = os.path.join(RESULTS_DIR, relative_path)

        if not os.path.isfile(csv_path):
            print(f"  [SKIP] {model_name}: {csv_path} non trovato.")
            continue

        df = pd.read_csv(csv_path)

        if "patient_id" not in df.columns:
            print(f"  [SKIP] {model_name}: manca la colonna patient_id nel CSV.")
            continue

        slice_level = evaluate(df)
        patient_df = compute_patient_level(df)
        patient_data_dict[model_name] = patient_df

        patient_level = {
            "auroc": roc_auc_score(patient_df["patient_label"], patient_df["patient_score"]),
            "ap": average_precision_score(patient_df["patient_label"], patient_df["patient_score"]),
        }

        rows.append({
            "model": model_name,
            "n_slices": len(df),
            "n_patients": len(patient_df),
            "slice_auroc": slice_level["auroc"],
            "slice_ap": slice_level["ap"],
            "patient_auroc": patient_level["auroc"],
            "patient_ap": patient_level["ap"],
        })

        print(
            f"  [OK]   {model_name:18s} | "
            f"Slice AUROC = {slice_level['auroc']:.4f}  --->  "
            f"Patient AUROC = {patient_level['auroc']:.4f} "
            f"({len(patient_df)} pazienti)"
        )

    if not rows:
        print("\nNessun CSV compatibile trovato.")
        return None

    result_df = pd.DataFrame(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "patient_level_comparison.csv")
    result_df.to_csv(out_csv, index=False)
    print(f"\n✓ Salvato CSV: {out_csv}")

    # ============================================================
    # PLOTS 1 & 2: Slice-level vs Patient-level (AUROC & AP)
    # ============================================================
    
    plot_comparison_bar(
        result_df, 
        "slice_auroc", "patient_auroc", 
        "AUROC: Confronto Slice-Level vs Patient-Level", 
        "patient_vs_slice_auroc.png"
    )
    print(f"✓ Salvato Grafico: patient_vs_slice_auroc.png")

    plot_comparison_bar(
        result_df, 
        "slice_ap", "patient_ap", 
        "Average Precision: Confronto Slice-Level vs Patient-Level", 
        "patient_vs_slice_ap.png",
        ylabel="AP Score"
    )
    print(f"✓ Salvato Grafico: patient_vs_slice_ap.png")

    # ============================================================
    # PLOT 3: Patient-Level ROC Curves (Multi-model)
    # ============================================================
    plt.figure(figsize=(7, 7))

    for model_name in ORDER_MODELS:
        if model_name in patient_data_dict:
            p_df = patient_data_dict[model_name]
            fpr, tpr, _ = roc_curve(p_df["patient_label"], p_df["patient_score"])
            auc_val = roc_auc_score(p_df["patient_label"], p_df["patient_score"])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", 
                     color=MODEL_COLORS.get(model_name, None), linewidth=2.5)

    plt.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1.5, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontweight='bold')
    plt.ylabel("True Positive Rate", fontweight='bold')
    plt.title("Curve ROC Patient-Level", fontweight="bold", pad=15, fontsize=14)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()

    fig_path_3 = os.path.join(OUT_DIR, "patient_level_roc_curves.png")
    plt.savefig(fig_path_3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvato Grafico: patient_level_roc_curves.png")

    print(f"\nTutti i risultati e i grafici patient-level sono completi in '{OUT_DIR}'!")
    return result_df

if __name__ == "__main__":
    run()