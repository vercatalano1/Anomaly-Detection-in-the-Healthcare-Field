# ============================================================
# FINAL THESIS PLOTS — COMPARATIVE ANALYSIS (Ordinamento Cronologico)
# ============================================================
# Legge automaticamente results/summary/model_comparison.csv
# e ordina i modelli nell'esatta sequenza in cui sono stati testati:
# Isolation Forest -> CNN Autoencoder -> CutPaste -> PatchCore.
# ============================================================

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURAZIONE E PATH
# ============================================================
# Risaliamo l'albero partendo da dove si trova questo script
current_path = Path(__file__).resolve()

# Cerchiamo la cartella "src" per capire dov'è la radice
while current_path.name != "src" and current_path.parent != current_path:
    current_path = current_path.parent

# La radice del progetto è la cartella "padre" di "src"
if current_path.name == "src":
    PROJECT_ROOT = current_path.parent
else:
    # Fallback generico (adatta se lo script è a 2 livelli di profondità)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_CSV = RESULTS_DIR / "summary" / "model_comparison.csv"

OUT_DIR = RESULTS_DIR / "summary" / "final_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# Stile accademico per i grafici
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.facecolor"] = "white"


def load_summary() -> pd.DataFrame:
    if not os.path.isfile(SUMMARY_CSV):
        print(
            f"ERRORE: non trovo {SUMMARY_CSV}.\n"
            "Esegui prima: python src/analysis/collect_results.py"
        )
        sys.exit(1)

    df = pd.read_csv(SUMMARY_CSV)

    # Per ogni modello, teniamo la variante migliore (o quella di riferimento)
    best_per_model = (
        df.sort_values("image_auroc", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )

    # Ordine di test stabilito: IF -> CNN Autoencoder -> CutPaste -> PatchCore
    order_mapping = {
        "Isolation Forest": 0,
        "CNN Autoencoder": 1,
        "CutPaste": 2,
        "PatchCore": 3
    }
    
    best_per_model["sort_idx"] = best_per_model["model"].map(order_mapping)
    return best_per_model.sort_values("sort_idx").drop(columns=["sort_idx"])


def short_label(model_name: str) -> str:
    labels = {
        "Isolation Forest": "Isolation Forest\n(ML Baseline)",
        "CNN Autoencoder": "CNN Autoencoder\n(Generative)",
        "CutPaste": "CutPaste\n(Self-Supervised)",
        "PatchCore": "PatchCore\n(Transfer Learning)",
    }
    return labels.get(model_name, model_name)


def annotate_bars(ax):
    for p in ax.patches:
        height = p.get_height()
        if height and height > 0 and not np.isnan(height):
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                xytext=(0, 5), textcoords="offset points"
            )


# ============================================================
# 1. GRAFICO IMAGE-LEVEL (DETECTION)
# ============================================================
def plot_image_level(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["Model"] = plot_df["model"].apply(short_label)

    melted = plot_df.melt(
        id_vars="Model",
        value_vars=["image_auroc", "image_ap"],
        var_name="Metric",
        value_name="Score"
    )

    melted["Metric"] = melted["Metric"].map({
        "image_auroc": "Image AUROC",
        "image_ap": "Image AP",
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Blues_d")
    
    annotate_bars(ax)

    plt.ylim(0, 1.1)
    plt.title("Image-Level Anomaly Detection (Triage Capability)", fontweight="bold", pad=20)
    plt.ylabel("Score (0 to 1)")
    plt.xlabel("")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "1_image_level_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================
# 2. GRAFICO PIXEL-LEVEL (LOCALIZATION)
# ============================================================
def plot_pixel_level(df: pd.DataFrame) -> None:
    plot_df = df[df["model"] != "Isolation Forest"].copy()
    plot_df["Model"] = plot_df["model"].apply(short_label)

    melted = plot_df.melt(
        id_vars="Model",
        value_vars=["pixel_auroc", "pixel_dice"],
        var_name="Metric",
        value_name="Score"
    )

    melted["Metric"] = melted["Metric"].map({
        "pixel_auroc": "Pixel AUROC",
        "pixel_dice": "Dice Score",
    })

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Greens_d")
    
    annotate_bars(ax)

    plt.ylim(0, 1.1)
    plt.title("Pixel-Level Localization (Segmentation Accuracy)", fontweight="bold", pad=20)
    plt.ylabel("Score (0 to 1)")
    plt.xlabel("")
    
    # <-- MODIFICA QUI: Spostata a sinistra all'interno del grafico -->
    plt.legend(loc="upper left")
    
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "2_pixel_level_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================
# 3. SCATTER PLOT (TRADE-OFF CON ZONE)
# ============================================================
def plot_tradeoff(df: pd.DataFrame) -> None:
    plot_df = df[df["model"] != "Isolation Forest"].copy()
    plot_df["Model"] = plot_df["model"].apply(short_label)

    if plot_df["pixel_dice"].isna().all():
        print("  [SKIP] tradeoff scatter: nessun dato Dice disponibile")
        return

    plt.figure(figsize=(8, 6))
    
    sns.scatterplot(
        data=plot_df, x="pixel_dice", y="image_auroc", hue="Model", 
        s=300, palette=["#2ca02c", "#e74c3c", "#1f77b4"], style="Model", markers=["o", "X", "s"]
    )
    
    plt.axhline(0.80, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.30, color='gray', linestyle='--', alpha=0.5)
    
    plt.text(0.05, 0.95, "Optimal Detection Zone", color="blue", fontsize=10, alpha=0.7, fontweight="bold")
    plt.text(0.40, 0.62, "Optimal Localization Zone", color="green", fontsize=10, alpha=0.7, fontweight="bold")
    plt.text(0.06, 0.60, "Domain Shift Failure", color="red", fontsize=10, alpha=0.7, fontweight="bold")

    plt.xlim(0.0, 0.55)
    plt.ylim(0.55, 1.0)
    plt.title("Detection vs. Localization Trade-off", fontweight="bold", pad=15)
    plt.xlabel("Localization Accuracy (Dice Score)")
    plt.ylabel("Detection Capability (Image AUROC)")
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title="Architecture")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "3_tradeoff_scatter.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================
# 4. METRICHE ALLA SOGLIA OPERATIVA
# ============================================================
def plot_threshold_metrics(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["Model"] = plot_df["model"].apply(short_label)

    value_cols = [c for c in ["image_sensitivity", "image_specificity", "image_f1"] if c in plot_df.columns]
    if not value_cols:
        return

    melted = plot_df.melt(
        id_vars="Model", value_vars=value_cols,
        var_name="Metric", value_name="Score"
    )

    melted["Metric"] = melted["Metric"].map({
        "image_sensitivity": "Sensitivity",
        "image_specificity": "Specificity",
        "image_f1": "F1",
    })

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Oranges_d")
    
    annotate_bars(ax)

    plt.ylim(0, 1.1)
    plt.title("Metriche alla soglia operativa (Image-Level)", fontweight="bold", pad=20)
    plt.ylabel("Score (0 to 1)")
    plt.xlabel("")
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "4_threshold_metrics_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================
# 5. COSTO COMPUTAZIONALE
# ============================================================
def plot_computational_cost(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["training_time_s"]).copy()
    if plot_df.empty:
        return

    plot_df["Model"] = plot_df["model"].apply(short_label)

    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=plot_df, x="Model", y="training_time_s", color="#4C72B0")
    ax.set_yscale("log")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{height:.1f}s",
            (p.get_x() + p.get_width() / 2., height),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            xytext=(0, 5), textcoords="offset points"
        )

    plt.title("Costo computazionale (training/esecuzione, scala log)", fontweight="bold", pad=20)
    plt.ylabel("Tempo (secondi, log scale)")
    plt.xlabel("")
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "5_computational_cost.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")


# ============================================================
# ESECUZIONE PRINCIPALE
# ============================================================
if __name__ == "__main__":
    print("Generating final thesis plots dynamically with custom order (IF -> CNN-AE -> CutPaste -> PatchCore)...")
    
    summary = load_summary()

    plot_image_level(summary)
    plot_pixel_level(summary)
    plot_tradeoff(summary)
    plot_threshold_metrics(summary)
    plot_computational_cost(summary)

    print(f"\nAll plots generated successfully in '{OUT_DIR}'!")