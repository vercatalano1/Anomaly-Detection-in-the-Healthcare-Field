# ============================================================
# COMPARE CNN-AE VARIANTS
# ============================================================
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# RICERCA ROBUSTA DELLA DIRECTORY RESULTS
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
    # Fallback generico
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "cnn_compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.facecolor"] = "white"

# ============================================================
# VARIANTI
# ============================================================
VARIANTS: List[Tuple[str, str]] = [
    ("Baseline (MSE)", "cnn_autoencoder"),
    ("Ablation: MSE", "cnn_autoencoder2_nofc_mse"),
    ("Ablation: L1", "cnn_autoencoder2_nofc_l1"),
    ("Ablation: MSE+L1", "cnn_autoencoder2_nofc_mse_l1"),
    ("+ Denoising (Gauss)", "cnn_autoencoder2_nofc_mse_l1_denoising_gaussiano"),
    ("+ Denoising (Spaz)", "cnn_autoencoder2_nofc_mse_l1_denoising_spaziale"),
    ("+ Denoising + PP", "cnn_autoencoder2_nofc_mse_l1_pp"),
]

def load_metrics() -> pd.DataFrame:
    rows = []
    for label, folder in VARIANTS:
        metrics_path = RESULTS_DIR / folder / "cnn_autoencoder_metrics.csv"

        if not metrics_path.is_file():
            print(f"  [SKIP] {label}: {metrics_path} non trovato")
            continue

        df = pd.read_csv(metrics_path)
        row = df.iloc[0]

        rows.append({
            "variant": label,
            "image_auroc": row.get("Image_AUROC"),
            "pixel_auroc": row.get("Pixel_AUROC"),
            "dice": row.get("Pixel_Dice"),
        })
        print(f"  [OK]   {label}")

    return pd.DataFrame(rows)

def plot_metrics_comparison(df: pd.DataFrame) -> None:
    melted = df.melt(
        id_vars="variant",
        value_vars=["image_auroc", "pixel_auroc", "dice"],
        var_name="Metric", value_name="Score"
    )

    melted["Metric"] = melted["Metric"].map({
        "image_auroc": "Image AUROC",
        "pixel_auroc": "Pixel AUROC",
        "dice": "Dice",
    })

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=melted, x="variant", y="Score", hue="Metric", palette="viridis")

    for p in ax.patches:
        height = p.get_height()
        if height and pd.notna(height) and height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        xytext=(0, 4), textcoords="offset points")

    plt.ylim(0, 1.1)
    plt.xticks(rotation=20, ha="right", fontweight="bold")
    plt.title("CNN Autoencoder — Ablation Study", fontweight="bold", pad=15)
    plt.xlabel("")
    plt.ylabel("Score (0 to 1)")
    plt.legend(loc="upper left")
    plt.tight_layout()

    path = OUT_DIR / "cnn_ae_variants_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

def plot_training_curves() -> None:
    plt.figure(figsize=(10, 6))
    n_curves_plotted = 0

    for label, folder in VARIANTS:
        history_path = RESULTS_DIR / folder / "training_history.csv"

        if not history_path.is_file():
            continue

        history = pd.read_csv(history_path)
        plt.plot(history["epoch"], history["val_loss"], label=label, linewidth=2)
        n_curves_plotted += 1

    if n_curves_plotted == 0:
        print("  [SKIP] training curves: nessun training_history.csv trovato")
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("CNN Autoencoder — Curve di validazione per variante", fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path = OUT_DIR / "cnn_ae_training_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

if __name__ == "__main__":
    print("=" * 70)
    print("CONFRONTO VARIANTI CNN AUTOENCODER")
    print("=" * 70)
    
    print(f"📁 Root del progetto individuata: {PROJECT_ROOT}")
    print(f"📁 Cartella Results individuata:  {RESULTS_DIR}")
    print("-" * 70)

    metrics_df = load_metrics()

    if metrics_df.empty:
        print("\nNessuna variante trovata. Controlla i percorsi stampati sopra.")
    else:
        out_csv = OUT_DIR / "cnn_ae_variants_metrics.csv"
        metrics_df.to_csv(out_csv, index=False)
        print(f"\n✓ Salvato CSV riassuntivo: {out_csv}")

        plot_metrics_comparison(metrics_df)
        plot_training_curves()
        print("\nElaborazione completata con successo!")