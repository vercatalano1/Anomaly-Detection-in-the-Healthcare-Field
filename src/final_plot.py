# ============================================================
# FINAL PLOT SCRIPT (Versione Definitiva e Ordinata)
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "results/summary/model_comparison.csv"
PATIENT_CSV = "results/summary/patient_level/patient_level_comparison.csv"
OUTPUT_DIR = "results/summary/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_ORDER = [
    "Isolation Forest",
    "CNN Autoencoder",
    "PatchCore"
]

MODEL_COLORS = {
    "Isolation Forest": "#d62728",
    "CNN Autoencoder": "#2ca02c",
    "PatchCore": "#1f77b4"
}

ROC_SOURCES = [
    ("Isolation Forest", "results/ml_baseline/error_analysis.csv"),
    ("CNN Autoencoder", "results/cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"),
    ("PatchCore", "results/patchcore/image_level_results.csv"),
]

plt.rcParams.update({"font.size": 11})


# ============================================================
# 1. IMAGE-LEVEL PERFORMANCE
# ============================================================
def plot_image_level_performance(plot_df):
    print("\n[1/7] Generazione Image-Level Performance...")
    image_metrics = [
        ("AUROC", "image_auroc"),
        ("Average Precision", "image_ap"),
        ("F1-Score", "image_f1"),
        ("Sensitivity", "image_sensitivity"),
        ("Specificity", "image_specificity"),
        ("Balanced Accuracy", "image_bacc")
    ]

    x = np.arange(len(plot_df))
    n_metrics = len(image_metrics)
    width = 0.80 / n_metrics

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric_name, column) in enumerate(image_metrics):
        values = plot_df[column].values
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], fontweight='bold')
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_title("Prestazioni Image-Level", fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "image_level_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 2. PIXEL-LEVEL PERFORMANCE
# ============================================================
def plot_pixel_level_performance(plot_df):
    print("[2/7] Generazione Pixel-Level Performance...")
    pixel_metrics = [
        ("AUROC", "pixel_auroc"),
        ("Average Precision", "pixel_ap"),
        ("Dice", "pixel_dice"),
        ("IoU", "pixel_iou"),
        ("Sensitivity", "pixel_sensitivity"),
        ("Specificity", "pixel_specificity")
    ]

    pixel_df = plot_df[plot_df["model"].isin(["CNN Autoencoder", "PatchCore"])].copy()
    x = np.arange(len(pixel_df))
    n_metrics = len(pixel_metrics)
    width = 0.80 / n_metrics

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (metric_name, column) in enumerate(pixel_metrics):
        values = pixel_df[column].values
        offset = (i - (n_metrics - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=metric_name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(pixel_df["model"], fontweight='bold')
    ax.set_ylabel("Score", fontweight='bold')
    ax.set_ylim(0, 1.08)
    ax.set_title("Prestazioni di Localizzazione Pixel-Level", fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "pixel_level_performance.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 3. SLICE-LEVEL ROC CURVES
# ============================================================
def plot_slice_level_roc():
    print("[3/7] Generazione Slice-Level ROC Curves...")
    fig, ax = plt.subplots(figsize=(7, 7))

    for model_name, csv_path in ROC_SOURCES:
        if not os.path.exists(csv_path):
            continue
        roc_df = pd.read_csv(csv_path)
        if "true_label" not in roc_df.columns or "anomaly_score" not in roc_df.columns:
            continue

        valid = roc_df["true_label"].notna() & roc_df["anomaly_score"].notna()
        y_true = roc_df.loc[valid, "true_label"].to_numpy()
        y_score = roc_df.loc[valid, "anomaly_score"].to_numpy()

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)

        ax.plot(fpr, tpr, linewidth=2.5, color=MODEL_COLORS[model_name], label=f"{model_name} (AUROC = {auc_value:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1.5, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontweight='bold')
    ax.set_ylabel("True Positive Rate", fontweight='bold')
    ax.set_title("Curve ROC Slice-Level", fontweight="bold", pad=15, fontsize=13)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "slice_level_roc_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 4. COMPUTATIONAL COST
# ============================================================
def plot_computational_cost(plot_df):
    print("[4/7] Generazione Computational Cost...")
    fig, ax = plt.subplots(figsize=(7, 5))
    times = plot_df["training_time_s"].values
    models = plot_df["model"].values
    colors = [MODEL_COLORS.get(m, "gray") for m in models]

    bars = ax.bar(models, times, color=colors, edgecolor='white', linewidth=1.5, width=0.5)

    for bar, t in zip(bars, times):
        if pd.notna(t):
            text = f"{t/60:.1f} min" if t > 60 else f"{t:.1f} s"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, text, 
                    ha='center', va='bottom', fontweight='bold', color='#333333')

    ax.set_ylabel("Tempo (secondi)", fontweight='bold')
    ax.set_title("Costo Computazionale", fontweight="bold", pad=15, fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "training_computational_cost.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 5. TRADE-OFF SCATTER PLOT
# ============================================================
def plot_tradeoff(plot_df):
    print("[5/7] Generazione Trade-off Scatter Plot...")
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in plot_df.iterrows():
        if pd.isna(row.get("pixel_dice")):
            continue
        color = MODEL_COLORS.get(row["model"], "gray")
        ax.scatter(row["pixel_dice"], row["image_auroc"], s=250, color=color, edgecolor='white', linewidth=1.5, label=row["model"], zorder=3)
        ax.annotate(row["model"], (row["pixel_dice"], row["image_auroc"]), xytext=(0, 15), textcoords="offset points", ha='center', fontweight='bold', color=color)

    ax.set_xlabel("Pixel-Level Dice (Localizzazione)", fontweight='bold')
    ax.set_ylabel("Image-Level AUROC (Classificazione)", fontweight='bold')
    ax.set_title("Trade-off: Classificazione vs Localizzazione", fontweight="bold", pad=15, fontsize=13)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "detection_localization_tradeoff.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 6. PRECISION-RECALL CURVES OVERLAY
# ============================================================
def plot_slice_level_pr_curves():
    print("[6/7] Generazione Precision-Recall Curves...")
    fig, ax = plt.subplots(figsize=(7, 7))

    for model_name, csv_path in ROC_SOURCES:
        if not os.path.exists(csv_path):
            continue
        df_model = pd.read_csv(csv_path)
        if "true_label" not in df_model.columns or "anomaly_score" not in df_model.columns:
            continue

        y_true = df_model["true_label"].to_numpy()
        scores = df_model["anomaly_score"].to_numpy()

        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap_value = average_precision_score(y_true, scores)

        ax.plot(recall, precision, linewidth=2.5, color=MODEL_COLORS[model_name], label=f"{model_name} (AP = {ap_value:.3f})")

    if os.path.exists(ROC_SOURCES[0][1]):
        sample_df = pd.read_csv(ROC_SOURCES[0][1])
        baseline = np.mean(sample_df["true_label"].to_numpy())
        ax.axhline(baseline, linestyle="--", color="#333333", linewidth=1.5, alpha=0.6, label=f"Baseline (Prevalence = {baseline:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.6, 1.05])
    ax.set_xlabel("Recall", fontweight='bold')
    ax.set_ylabel("Precision", fontweight='bold')
    ax.set_title("Curve Precision-Recall Slice-Level", fontweight="bold", pad=15, fontsize=13)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "slice_level_pr_curves.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# 7. PIXEL-LEVEL DICE DISTRIBUTION (BOXPLOT)
# ============================================================
def post_process_mask(anomaly_map, threshold, min_size=20):
    binary_mask = anomaly_map >= threshold
    labeled_mask, _ = ndimage.label(binary_mask)
    component_sizes = np.bincount(labeled_mask.ravel())
    keep = component_sizes >= min_size
    keep[0] = False
    binary_mask = keep[labeled_mask]
    binary_mask = ndimage.binary_closing(binary_mask, structure=np.ones((3, 3)))
    return binary_mask.astype(np.uint8)

def plot_dice_boxplot():
    print("[7/7] Generazione Boxplot Comparativo Dice (Per-slice)...")
    from data_analysis.dataloader import get_dataset
    import torch

    test_ds = get_dataset("brats", img_size=64, mode="test")
    tumor_indices = []
    for i in range(len(test_ds)):
        mask = test_ds[i]["mask"]
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if np.sum(mask > 0) > 0:
            tumor_indices.append(i)

    models_info = {
        "CNN Autoencoder": {"dir": "results/cnn_autoencoder2_nofc_mse_l1_pp/anomaly_maps", "threshold": 0.124},
        "PatchCore": {"dir": "results/patchcore/anomaly_maps", "threshold": 1.154}
    }

    dice_results = {}
    for model_name, info in models_info.items():
        maps_dir, thresh = info["dir"], info["threshold"]
        model_dices = []
        for idx in tumor_indices:
            sample = test_ds[idx]
            mask = sample["mask"]
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            true_mask = (np.squeeze(mask) > 0).astype(np.uint8)

            map_path = os.path.join(maps_dir, f"anomaly_map_{idx:05d}.npy")
            if not os.path.exists(map_path):
                continue
            anomaly_map = np.load(map_path)
            pred_mask = post_process_mask(anomaly_map, thresh)

            intersection = np.logical_and(pred_mask, true_mask).sum()
            sum_pixels = pred_mask.sum() + true_mask.sum()
            dice = 1.0 if sum_pixels == 0 else float(2.0 * intersection / sum_pixels)
            model_dices.append(dice)
        dice_results[model_name] = model_dices

    fig, ax = plt.subplots(figsize=(7, 6))
    models_list = ["CNN Autoencoder", "PatchCore"]
    data_to_plot = [dice_results[m] for m in models_list]
    colors_list = [MODEL_COLORS[m] for m in models_list]

    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.4,
                    boxprops=dict(linewidth=1.5), medianprops=dict(color='black', linewidth=2))

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, data in enumerate(data_to_plot):
        y = data
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax.scatter(x, y, alpha=0.3, color=colors_list[i], edgecolor='none', s=20, zorder=3)

    ax.set_xticklabels(models_list, fontweight='bold', fontsize=11)
    ax.set_ylabel("Dice Similarity Coefficient (per-slice)", fontweight='bold')
    ax.set_title("Distribuzione del Dice Score sulle Lesioni Tumorali", fontweight="bold", pad=15, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "pixel_dice_boxplot.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Salvato: {output_path}")


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================
def main():
    print("=" * 70)
    print("GENERAZIONE COMPLETA FIGURE FINALI")
    print("=" * 70)

    if not os.path.exists(INPUT_CSV):
        print(f"[ERRORE] File di input non trovato: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    plot_df = df[df["model"].isin(MODEL_ORDER)].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=MODEL_ORDER, ordered=True)
    plot_df = plot_df.sort_values("model").reset_index(drop=True)

    # Esecuzione ordinata delle 7 funzioni di plot
    plot_image_level_performance(plot_df)
    plot_pixel_level_performance(plot_df)
    plot_slice_level_roc()
    plot_computational_cost(plot_df)
    plot_tradeoff(plot_df)
    plot_slice_level_pr_curves()
    plot_dice_boxplot()

    print("\n" + "=" * 70)
    print("TUTTE LE FIGURE FINALI SONO STATE GENERATE CON SUCCESSO!")
    print(f"Cartella di output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
