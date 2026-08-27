# ============================================================
# FINAL THESIS PLOTS — COMPARATIVE ANALYSIS
# ============================================================
# Questo script genera i grafici a barre e scatter plot
# per il capitolo dei Risultati/Conclusioni della tesi.
# Utilizza i dati esatti estratti dai report dei 4 modelli.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIGURAZIONE
# ============================================================
OUT_DIR = os.path.join("results", "final_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# Stile accademico per i grafici
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.facecolor"] = "white"

# ============================================================
# DATI ESTRATTI DAI REPORT DELLA TESI
# ============================================================
data = [
    {
        "Model": "Isolation Forest\n(ML Baseline)",
        "Image AUROC": 0.6704,
        "Image AP": 0.7935,
        "Pixel AUROC": 0.0000, # N/A per IF
        "Dice Score": 0.0000   # N/A per IF
    },
    {
        "Model": "CNN Autoencoder\n(Generative)",
        "Image AUROC": 0.6723,
        "Image AP": 0.8187,
        "Pixel AUROC": 0.9189,
        "Dice Score": 0.4778
    },
    {
        "Model": "CutPaste\n(Self-Supervised)",
        "Image AUROC": 0.6178,
        "Image AP": 0.7602,
        "Pixel AUROC": 0.7241,
        "Dice Score": 0.0524
    },
    {
        "Model": "PatchCore\n(Transfer Learning)",
        "Image AUROC": 0.9037,
        "Image AP": 0.9590,
        "Pixel AUROC": 0.9561,
        "Dice Score": 0.3127
    }
]

df = pd.DataFrame(data)

# Colori per i modelli
COLORS = ["#7f8c8d", "#2ca02c", "#e74c3c", "#1f77b4"]

# ============================================================
# 1. GRAFICO IMAGE-LEVEL (DETECTION)
# ============================================================
def plot_image_level():
    df_melted = df.melt(id_vars="Model", value_vars=["Image AUROC", "Image AP"], 
                        var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Blues_d")
    
    # Aggiungi i numeri sopra le barre
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 5), 
                        textcoords='offset points')

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
def plot_pixel_level():
    # Rimuoviamo Isolation Forest perché non fa pixel-level
    df_pixel = df[df["Model"] != "Isolation Forest\n(ML Baseline)"]
    df_melted = df_pixel.melt(id_vars="Model", value_vars=["Pixel AUROC", "Dice Score"], 
                              var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Greens_d")
    
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}", 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=11, fontweight='bold', xytext=(0, 5), 
                    textcoords='offset points')

    plt.ylim(0, 1.1)
    plt.title("Pixel-Level Localization (Segmentation Accuracy)", fontweight="bold", pad=20)
    plt.ylabel("Score (0 to 1)")
    plt.xlabel("")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "2_pixel_level_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {path}")

# ============================================================
# 3. SCATTER PLOT (TRADE-OFF)
# ============================================================
def plot_tradeoff():
    # Solo i 3 modelli Deep Learning
    df_dl = df[df["Model"] != "Isolation Forest\n(ML Baseline)"]
    
    plt.figure(figsize=(8, 6))
    
    # Plot dei punti
    sns.scatterplot(data=df_dl, x="Dice Score", y="Image AUROC", hue="Model", 
                    s=300, palette=["#2ca02c", "#e74c3c", "#1f77b4"], style="Model", markers=["o", "X", "s"])
    
    # Linee tratteggiate di riferimento
    plt.axhline(0.80, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.30, color='gray', linestyle='--', alpha=0.5)
    
    # Annotazioni per le "Zone" del grafico
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
# ESECUZIONE
# ============================================================
if __name__ == "__main__":
    print("Generating final thesis plots...")
    plot_image_level()
    plot_pixel_level()
    plot_tradeoff()
    print("\nAll plots generated successfully in 'results/final_comparison/'!")