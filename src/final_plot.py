
# ============================================================
# FINAL VISUALIZATION SCRIPT
# Comparison: Isolation Forest vs CNN Autoencoder vs PatchCore
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "results/summary/model_comparison.csv"
OUTPUT_DIR = "results/summary/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("\nLoaded results:")
print(df.to_string(index=False))


# ============================================================
# SELECT BEST VARIANT PER MODEL
# ============================================================
# If multiple variants are present, select the one with the
# highest image-level AUROC.

best_per_model = (
    df.sort_values("image_auroc", ascending=False)
      .groupby("model", as_index=False)
      .first()
)


# Fixed order for final thesis figures
model_order = [
    "Isolation Forest",
    "CNN Autoencoder",
    "PatchCore"
]

best_per_model["model"] = pd.Categorical(
    best_per_model["model"],
    categories=model_order,
    ordered=True
)

plot_df = (
    best_per_model
    .sort_values("model")
    .reset_index(drop=True)
)

print("\nSelected final models:")
print(
    plot_df[
        [
            "model",
            "variant",
            "image_auroc",
            "image_ap",
            "image_f1",
            "image_sensitivity",
            "image_specificity",
            "pixel_auroc",
            "pixel_dice",
            "training_time_s"
        ]
    ].to_string(index=False)
)


# ============================================================
# COMMON SETTINGS
# ============================================================

models = plot_df["model"].tolist()
x = np.arange(len(models))

FIGSIZE = (8, 5)
DPI = 300


# ============================================================
# 1. IMAGE-LEVEL AUROC + AP
# ============================================================

fig, ax = plt.subplots(figsize=FIGSIZE)

width = 0.35

ax.bar(
    x - width / 2,
    plot_df["image_auroc"],
    width,
    label="AUROC"
)

ax.bar(
    x + width / 2,
    plot_df["image_ap"],
    width,
    label="Average Precision"
)

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Image-Level Detection Performance")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "image_level_auroc_ap.png"),
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. PIXEL-LEVEL AUROC + DICE
# ============================================================
# Isolation Forest does not provide pixel-level localization,
# therefore it is excluded from this comparison.

pixel_df = plot_df[
    plot_df["model"].isin([
        "CNN Autoencoder",
        "PatchCore"
    ])
].copy()

fig, ax = plt.subplots(figsize=FIGSIZE)

pixel_models = pixel_df["model"].tolist()
pixel_x = np.arange(len(pixel_models))

width = 0.35

ax.bar(
    pixel_x - width / 2,
    pixel_df["pixel_auroc"],
    width,
    label="AUROC"
)

ax.bar(
    pixel_x + width / 2,
    pixel_df["pixel_dice"],
    width,
    label="Dice"
)

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Pixel-Level Localization Performance")
ax.set_xticks(pixel_x)
ax.set_xticklabels(pixel_models, rotation=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "pixel_level_auroc_dice.png"),
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 3. DETECTION vs LOCALIZATION TRADE-OFF
# ============================================================
# No arbitrary performance zones are introduced.

fig, ax = plt.subplots(figsize=FIGSIZE)

for _, row in plot_df.iterrows():

    # Isolation Forest has no pixel-level Dice
    if pd.isna(row["pixel_dice"]):
        continue

    ax.scatter(
        row["pixel_dice"],
        row["image_auroc"],
        s=100,
        label=row["model"]
    )

    ax.annotate(
        row["model"],
        (
            row["pixel_dice"],
            row["image_auroc"]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

ax.set_xlabel("Pixel-Level Dice")
ax.set_ylabel("Image-Level AUROC")
ax.set_title("Detection vs Localization Trade-off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "detection_localization_tradeoff.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 4. IMAGE-LEVEL THRESHOLD-BASED METRICS
# ============================================================

metric_columns = [
    "image_sensitivity",
    "image_specificity",
    "image_f1"
]

metric_labels = {
    "image_sensitivity": "Sensitivity",
    "image_specificity": "Specificity",
    "image_f1": "F1"
}

fig, ax = plt.subplots(figsize=FIGSIZE)

width = 0.25

for i, col in enumerate(metric_columns):

    ax.bar(
        x + (i - 1) * width,
        plot_df[col],
        width,
        label=metric_labels[col]
    )

ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_title("Image-Level Threshold-Based Performance")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "image_level_threshold_metrics.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 5. COMPUTATIONAL COST
# ============================================================
# Training time only.
#
# Note:
# Isolation Forest, CNN-AE and PatchCore training times are
# compared as reported in the summary CSV.

fig, ax = plt.subplots(figsize=FIGSIZE)

training_times = plot_df["training_time_s"]

ax.bar(
    x,
    training_times
)

ax.set_ylabel("Training Time (seconds)")
ax.set_xlabel("Model")
ax.set_title("Training Computational Cost")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)

# Log scale is useful when training times differ substantially.
ax.set_yscale("log")

ax.grid(
    axis="y",
    alpha=0.3,
    which="both"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_computational_cost.png"
    ),
    dpi=DPI,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 6. SAVE FINAL PLOTTING DATA
# ============================================================
# Useful for reproducibility and for checking exactly which
# model variants were used in the final figures.

final_csv = os.path.join(
    OUTPUT_DIR,
    "final_models_used_for_figures.csv"
)

plot_df.to_csv(
    final_csv,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("FINAL VISUALIZATION COMPLETED")
print("=" * 60)

print("\nFigures saved in:")
print(f"  {OUTPUT_DIR}")

print("\nGenerated files:")
print("  - image_level_auroc_ap.png")
print("  - pixel_level_auroc_dice.png")
print("  - detection_localization_tradeoff.png")
print("  - image_level_threshold_metrics.png")
print("  - training_computational_cost.png")
print("  - final_models_used_for_figures.csv")

print("\nFinal model order:")
for model in model_order:
    if model in plot_df["model"].astype(str).values:
        print(f"  - {model}")

print("\nNo Balanced Accuracy was added.")
print("=" * 60)
