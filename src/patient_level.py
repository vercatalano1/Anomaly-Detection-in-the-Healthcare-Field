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
import seaborn as sns
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

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.facecolor"] = "white"


# ============================================================
# SORGENTI DEI MODELLI PRINCIPALI (Migliore variante per architettura)
# ============================================================

COMPATIBLE_SOURCES = [
    ("Isolation Forest", "ml_baseline/error_analysis.csv"),
    ("CNN Autoencoder", "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"),
    ("CutPaste", "cutpaste/image_level_results.csv"),
    ("PatchCore", "patchcore/image_level_results.csv"),
]

ORDER_MODELS = ["Isolation Forest", "CNN Autoencoder", "CutPaste", "PatchCore"]
MODEL_COLORS = {
    "Isolation Forest": "#7f8c8d",
    "CNN Autoencoder": "#2ca02c",
    "CutPaste": "#e74c3c",
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
        "auroc": roc_auc_score(df["true_label"] if "true_label" in df.columns else df["patient_label"],
                                df["anomaly_score"] if "anomaly_score" in df.columns else df["patient_score"]),
        "ap": average_precision_score(df["true_label"] if "true_label" in df.columns else df["patient_label"],
                                      df["anomaly_score"] if "anomaly_score" in df.columns else df["patient_score"]),
    }


def run() -> Optional[pd.DataFrame]:
    print("=" * 70)
    print("PATIENT-LEVEL ANALYSIS — Tutti i modelli")
    print("=" * 70)

    rows = []
    patient_data_dict = {}  # Per memorizzare i dataframe a livello di paziente per i plot avanzati

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
    # PLOT 1: Slice-level vs Patient-level AUROC (Barplot)
    # ============================================================
    melted = result_df.melt(
        id_vars="model", value_vars=["slice_auroc", "patient_auroc"],
        var_name="Livello", value_name="AUROC"
    )

    melted["Livello"] = melted["Livello"].map({
        "slice_auroc": "Slice-level",
        "patient_auroc": "Patient-level",
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=melted, x="model", y="AUROC", hue="Livello", 
        order=[m for m in ORDER_MODELS if m in melted["model"].values],
        palette="Purples_d"
    )

    for p in ax.patches:
        height = p.get_height()
        if height and not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        xytext=(0, 4), textcoords="offset points")

    plt.ylim(0, 1.1)
    plt.title("AUROC Comparison: Slice-Level vs Patient-Level", fontweight="bold", pad=15)
    plt.xlabel("")
    plt.legend(loc="upper left")
    plt.tight_layout()

    fig_path_1 = os.path.join(OUT_DIR, "patient_vs_slice_auroc.png")
    plt.savefig(fig_path_1, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvato Grafico: {fig_path_1}")

    # ============================================================
    # PLOT 2: Slice-level vs Patient-level Average Precision (Barplot)
    # ============================================================
    melted_ap = result_df.melt(
        id_vars="model", value_vars=["slice_ap", "patient_ap"],
        var_name="Livello", value_name="AP"
    )

    melted_ap["Livello"] = melted_ap["Livello"].map({
        "slice_ap": "Slice-level",
        "patient_ap": "Patient-level",
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=melted_ap, x="model", y="AP", hue="Livello", 
        order=[m for m in ORDER_MODELS if m in melted_ap["model"].values],
        palette="Blues_d"
    )

    for p in ax.patches:
        height = p.get_height()
        if height and not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        xytext=(0, 4), textcoords="offset points")

    plt.ylim(0, 1.1)
    plt.title("Average Precision Comparison: Slice-Level vs Patient-Level", fontweight="bold", pad=15)
    plt.xlabel("")
    plt.legend(loc="upper left")
    plt.tight_layout()

    fig_path_2 = os.path.join(OUT_DIR, "patient_vs_slice_ap.png")
    plt.savefig(fig_path_2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvato Grafico: {fig_path_2}")

    # ============================================================
    # PLOT 3: Patient-Level ROC Curves (Multi-model)
    # ============================================================
    plt.figure(figsize=(8, 7))

    for model_name in ORDER_MODELS:
        if model_name in patient_data_dict:
            p_df = patient_data_dict[model_name]
            fpr, tpr, _ = roc_curve(p_df["patient_label"], p_df["patient_score"])
            auc_val = roc_auc_score(p_df["patient_label"], p_df["patient_score"])
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", 
                     color=MODEL_COLORS.get(model_name, None), linewidth=2.5)

    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Patient-Level ROC Curves", fontweight="bold", pad=15)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path_3 = os.path.join(OUT_DIR, "patient_level_roc_curves.png")
    plt.savefig(fig_path_3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Salvato Grafico: {fig_path_3}")

    print(f"\nTutti i risultati e i grafici patient-level sono completi in '{OUT_DIR}'!")
    return result_df


if __name__ == "__main__":
    run()