import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_analysis.dataloader import get_dataset


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 64

VISUALIZATION_INDICES = [
    828,
    1314,
    1801,
    2288,
    2775
]

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
    Raw anomaly maps are preserved for metrics and analysis.
    """

    anomaly_map = np.asarray(
        anomaly_map,
        dtype=np.float32
    )

    min_val = anomaly_map.min()
    max_val = anomaly_map.max()

    if max_val - min_val < 1e-8:
        return np.zeros_like(anomaly_map)

    return (
        (anomaly_map - min_val)
        / (max_val - min_val)
    )


# ============================================================
# PLOT COMPARISON
# ============================================================

def plot_comparison(index):

    index = int(index)

    # --------------------------------------------------------
    # Dataset sample
    # --------------------------------------------------------

    sample = test_ds[index]

    image = sample["img"]

    if torch.is_tensor(image):
        image = (
            image
            .squeeze()
            .cpu()
            .numpy()
        )

    mask = sample["mask"]

    if torch.is_tensor(mask):
        mask = (
            mask
            .squeeze()
            .cpu()
            .numpy()
        )

    mask = (
        mask > 0
    ).astype(np.float32)

    # --------------------------------------------------------
    # Load RAW anomaly maps
    # --------------------------------------------------------

    cnn_path = os.path.join(
        CNN_AE_DIR,
        f"anomaly_map_{index:05d}.npy"
    )

    patchcore_path = os.path.join(
        PATCHCORE_DIR,
        f"anomaly_map_{index:05d}.npy"
    )

    if not os.path.exists(cnn_path):
        raise FileNotFoundError(
            f"CNN-AE anomaly map non trovata:\n{cnn_path}"
        )

    if not os.path.exists(patchcore_path):
        raise FileNotFoundError(
            f"PatchCore anomaly map non trovata:\n{patchcore_path}"
        )

    cnn_map = np.load(cnn_path)
    patchcore_map = np.load(patchcore_path)

    # --------------------------------------------------------
    # Normalize ONLY for visualization
    # --------------------------------------------------------

    cnn_map_vis = normalize_map(
        cnn_map
    )

    patchcore_map_vis = normalize_map(
        patchcore_map
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 3.8)
    )

    # ========================================================
    # 1. ORIGINAL
    # ========================================================

    axes[0].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0].set_title(
        "Original"
    )

    # ========================================================
    # 2. GROUND TRUTH
    # ========================================================

    axes[1].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1].imshow(
        mask,
        cmap="Reds",
        alpha=0.75,
        vmin=0,
        vmax=1
    )

    axes[1].contour(
        mask,
        levels=[0.5],
        colors="cyan",
        linewidths=1
    )

    axes[1].set_title(
        "Ground Truth"
    )

    # ========================================================
    # 3. CNN-AE
    # ========================================================

    axes[2].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].imshow(
        cnn_map_vis,
        cmap="inferno",
        alpha=0.55,
        vmin=0,
        vmax=1
    )

    axes[2].contour(
        mask,
        levels=[0.5],
        colors="cyan",
        linewidths=1
    )

    axes[2].set_title(
        "CNN-AE"
    )

    # ========================================================
    # 4. PATCHCORE
    # ========================================================

    axes[3].imshow(
        image,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].imshow(
        patchcore_map_vis,
        cmap="inferno",
        alpha=0.55,
        vmin=0,
        vmax=1
    )

    axes[3].contour(
        mask,
        levels=[0.5],
        colors="cyan",
        linewidths=1
    )

    axes[3].set_title(
        "PatchCore"
    )

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Localization comparison | "
        f"index={index} | "
        f"patient={sample['patient_id']}",
        fontsize=12
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUT_DIR,
        f"comparison_{index:05d}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"  ✓ Saved: {output_path}"
    )


# ============================================================
# RUN
# ============================================================

print(
    "\nGenerating comparative heatmaps...\n"
)

for index in VISUALIZATION_INDICES:

    plot_comparison(index)

print(
    "\nDone."
)