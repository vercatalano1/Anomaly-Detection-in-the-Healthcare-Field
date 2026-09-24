import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_analysis.dataloader import get_dataset


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 64

VISUALIZATION_INDICES = [414, 1305, 2277, 2340, 2603, 2652]

CNN_AE_DIR = os.path.join(
    "results",
    "cnn_autoencoder2_nofc_mse_l1_pp",
    "anomaly_maps"
)

PATCHCORE_DIR = os.path.join(
    "results",
    "patchcore",
    "anomaly_maps"
)

OUT_DIR = os.path.join(
    "results",
    "summary",
    "heatmaps"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

test_ds = get_dataset(
    "brats",
    img_size=IMAGE_SIZE,
    mode="test"
)


# ============================================================
# NORMALIZATION FOR VISUALIZATION
# ============================================================

def normalize_map(anomaly_map):
    """
    Normalize each anomaly map independently to [0, 1].
    This normalization is used ONLY for visualization.
    """
    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32
    )

    min_val = anomaly_map.min()
    max_val = anomaly_map.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(anomaly_map)

    return (anomaly_map - min_val) / (max_val - min_val)


# ============================================================
# PLOT COMPARISON (HORIZONTAL GRID)
# ============================================================

def plot_all_comparisons():
    n_cols = len(VISUALIZATION_INDICES)  # 5 immagini
    n_rows = 4  # 4 tipologie (Original, GT, CNN, PatchCore)

    # Creiamo una figura orientata orizzontalmente
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 12)
    )

    row_labels = ["Immagine Originale", "Ground Truth", "CNN-AE", "PatchCore"]

    for col_idx, index in enumerate(VISUALIZATION_INDICES):
        index = int(index)

        # --------------------------------------------------------
        # Dataset sample
        # --------------------------------------------------------
        sample = test_ds[index]
        image = sample["img"]

        if torch.is_tensor(image):
            image = image.squeeze().cpu().numpy()

        mask = sample["mask"]
        if torch.is_tensor(mask):
            mask = mask.squeeze().cpu().numpy()

        mask = (mask > 0).astype(np.float32)

        # --------------------------------------------------------
        # Load RAW anomaly maps
        # --------------------------------------------------------
        cnn_path = os.path.join(CNN_AE_DIR, f"anomaly_map_{index:05d}.npy")
        patchcore_path = os.path.join(PATCHCORE_DIR, f"anomaly_map_{index:05d}.npy")

        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN-AE anomaly map non trovata:\n{cnn_path}")
        if not os.path.exists(patchcore_path):
            raise FileNotFoundError(f"PatchCore anomaly map non trovata:\n{patchcore_path}")

        cnn_map = np.load(cnn_path)
        patchcore_map = np.load(patchcore_path)

        # --------------------------------------------------------
        # Normalize ONLY for visualization
        # --------------------------------------------------------
        cnn_map_vis = normalize_map(cnn_map)
        patchcore_map_vis = normalize_map(patchcore_map)

        # --------------------------------------------------------
        # Assegnazione degli assi verticali per questa colonna
        # --------------------------------------------------------
        ax_orig = axes[0, col_idx]
        ax_gt = axes[1, col_idx]
        ax_cnn = axes[2, col_idx]
        ax_patch = axes[3, col_idx]

        # 1. ORIGINAL
        ax_orig.imshow(image, cmap="gray", vmin=0, vmax=1)
        # Titolo in cima a ogni colonna
        ax_orig.set_title(f"ID: {index}", fontsize=14, pad=10, fontweight="bold")

        # 2. GROUND TRUTH
        ax_gt.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_gt.imshow(mask, cmap="Reds", alpha=0.75, vmin=0, vmax=1)
        ax_gt.contour(mask, levels=[0.5], colors="cyan", linewidths=1)

        # 3. CNN-AE
        ax_cnn.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_cnn.imshow(cnn_map_vis, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        ax_cnn.contour(mask, levels=[0.5], colors="cyan", linewidths=1)

        # 4. PATCHCORE
        ax_patch.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax_patch.imshow(patchcore_map_vis, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        ax_patch.contour(mask, levels=[0.5], colors="cyan", linewidths=1)

    # --------------------------------------------------------
    # Formatting assi e label laterali
    # --------------------------------------------------------
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            
            # Etichette delle righe (solo sulla prima colonna a sinistra)
            if c == 0:
                axes[r, c].set_ylabel(row_labels[r], fontsize=14, labelpad=15, fontweight="bold")

    # Spaziatura generale
    fig.suptitle("Confronto di Localizzazione Pixel-Level", fontsize=18, fontweight="bold", y=0.98)
    
    # Riduce lo spazio vuoto tra le immagini rendendo la griglia compatta
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    output_path = os.path.join(OUT_DIR, "comparison_all_models_horizontal.png")
    
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"  ✓ Saved combined horizontal grid: {output_path}")

    plt.show()
    plt.close()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("\nGenerating comprehensive comparative horizontal heatmap grid...\n")
    plot_all_comparisons()
    print("\nDone.")


