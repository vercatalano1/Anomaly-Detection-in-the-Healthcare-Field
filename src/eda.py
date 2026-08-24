from pathlib import Path
from collections import defaultdict
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\catal\OneDrive\Desktop\Anomaly-Detection-in-the-Healthcare-Field\BraTS2021"
)

OUTPUT_DIR = Path("results") / "eda"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = OUTPUT_DIR / "eda_report.txt"

RANDOM_SEED = 42

# Numero di immagini utilizzate per alcune visualizzazioni
N_EXAMPLES = 6

# Numero massimo di immagini per classe
# utilizzate per alcune analisi computazionalmente
# più costose.
N_PCA = 1500
N_WASSERSTEIN = 200

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# OUTPUT TERMINALE + REPORT
# ============================================================

class Tee:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


report_file = open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(report_file)


# ============================================================
# FUNZIONI UTILI
# ============================================================

def section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


def extract_patient_id(filename):

    match = re.match(
        r"(BraTS2021_\d+)_",
        filename
    )

    if match:
        return match.group(1)

    return None


def extract_slice_number(filename):

    match = re.search(
        r"_(\d+)\.png$",
        filename
    )

    if match:
        return int(match.group(1))

    return None


def load_image(path):

    with Image.open(path) as img:

        img = img.convert("L")

        return np.asarray(img)


def save_figure(fig, filename):

    path = FIGURES_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"[FIGURA] {path}"
    )


def balanced_sample(files_a, files_b, n):

    n_a = min(n, len(files_a))
    n_b = min(n, len(files_b))

    selected_a = rng.choice(
        files_a,
        size=n_a,
        replace=False
    )

    selected_b = rng.choice(
        files_b,
        size=n_b,
        replace=False
    )

    return list(selected_a), list(selected_b)


def image_statistics(image):

    image = image.astype(np.float32)

    nonzero = image > 0

    return {

        "min": float(image.min()),

        "max": float(image.max()),

        "mean": float(image.mean()),

        "std": float(image.std()),

        "median": float(np.median(image)),

        "p01": float(np.percentile(image, 1)),

        "p05": float(np.percentile(image, 5)),

        "p25": float(np.percentile(image, 25)),

        "p75": float(np.percentile(image, 75)),

        "p95": float(np.percentile(image, 95)),

        "p99": float(np.percentile(image, 99)),

        "nonzero_fraction":
            float(nonzero.mean())
    }


def get_image_array(files):

    images = []

    for path in files:

        images.append(
            load_image(path).astype(
                np.float32
            )
        )

    return np.asarray(images)


# ============================================================
# CONTROLLO DATASET
# ============================================================

section("0. DATASET")

if not DATASET_DIR.exists():

    print(
        f"ERRORE: dataset non trovato:\n"
        f"{DATASET_DIR.resolve()}"
    )

    report_file.close()
    sys.stdout = sys.__stdout__

    raise SystemExit(1)


TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

NORMAL_DIR = TEST_DIR / "normal"
TUMOR_DIR = TEST_DIR / "tumor"
ANNOTATION_DIR = TEST_DIR / "annotation"


required_dirs = [
    TRAIN_DIR,
    NORMAL_DIR,
    TUMOR_DIR,
    ANNOTATION_DIR
]


for directory in required_dirs:

    if not directory.exists():

        print(
            f"ERRORE: cartella mancante:\n"
            f"{directory}"
        )

        report_file.close()
        sys.stdout = sys.__stdout__

        raise SystemExit(1)


train_files = sorted(
    TRAIN_DIR.glob("*.png")
)

normal_files = sorted(
    NORMAL_DIR.glob("*.png")
)

tumor_files = sorted(
    TUMOR_DIR.glob("*.png")
)

annotation_files = sorted(
    ANNOTATION_DIR.glob("*.png")
)


print(
    f"Dataset: {DATASET_DIR.resolve()}"
)

print(
    f"\nTrain:       {len(train_files)}"
)

print(
    f"Test normal: {len(normal_files)}"
)

print(
    f"Test tumor:  {len(tumor_files)}"
)

print(
    f"Annotation:  {len(annotation_files)}"
)


# ============================================================
# 1. PAZIENTI
# ============================================================

section("1. ANALISI DEI PAZIENTI")


def create_patient_dictionary(files):

    patients = defaultdict(list)

    for path in files:

        patient = extract_patient_id(
            path.name
        )

        if patient is not None:

            patients[patient].append(path)

    return patients


train_patients = create_patient_dictionary(
    train_files
)

normal_patients = create_patient_dictionary(
    normal_files
)

tumor_patients = create_patient_dictionary(
    tumor_files
)

annotation_patients = create_patient_dictionary(
    annotation_files
)


print(
    f"Pazienti TRAIN: "
    f"{len(train_patients)}"
)

print(
    f"Pazienti TEST NORMAL: "
    f"{len(normal_patients)}"
)

print(
    f"Pazienti TEST TUMOR: "
    f"{len(tumor_patients)}"
)

print(
    f"Pazienti con annotation: "
    f"{len(annotation_patients)}"
)


# ============================================================
# 2. DATA LEAKAGE
# ============================================================

section("2. CONTROLLO DATA LEAKAGE")


train_set = set(train_patients)
normal_set = set(normal_patients)
tumor_set = set(tumor_patients)


overlap_train_normal = (
    train_set & normal_set
)

overlap_train_tumor = (
    train_set & tumor_set
)

overlap_normal_tumor = (
    normal_set & tumor_set
)


print(
    f"TRAIN ∩ TEST NORMAL: "
    f"{len(overlap_train_normal)}"
)

print(
    f"TRAIN ∩ TEST TUMOR: "
    f"{len(overlap_train_tumor)}"
)

print(
    f"TEST NORMAL ∩ TEST TUMOR: "
    f"{len(overlap_normal_tumor)}"
)


if overlap_train_normal:

    print(
        "\nATTENZIONE: pazienti condivisi "
        "tra TRAIN e TEST NORMAL:"
    )

    print(
        sorted(overlap_train_normal)
    )


if overlap_train_tumor:

    print(
        "\nATTENZIONE: pazienti condivisi "
        "tra TRAIN e TEST TUMOR:"
    )

    print(
        sorted(overlap_train_tumor)
    )


# ============================================================
# 3. DISTRIBUZIONE CLASSI
# ============================================================

section("3. DISTRIBUZIONE DEL DATASET")


print(
    f"TRAIN:       {len(train_files)}"
)

print(
    f"TEST NORMAL: {len(normal_files)}"
)

print(
    f"TEST TUMOR:  {len(tumor_files)}"
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)

labels = [
    "Train",
    "Test Normal",
    "Test Tumor"
]

values = [
    len(train_files),
    len(normal_files),
    len(tumor_files)
]

bars = ax.bar(
    labels,
    values
)

ax.set_ylabel(
    "Numero di immagini"
)

ax.set_title(
    "Distribuzione delle immagini"
)

for bar, value in zip(
    bars,
    values
):

    ax.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )


save_figure(
    fig,
    "01_class_distribution.png"
)


# ============================================================
# 4. SLICE PER PAZIENTE
# ============================================================

section("4. SLICE PER PAZIENTE")


def patient_counts(patient_dict):

    return np.array(
        [
            len(v)
            for v in patient_dict.values()
        ]
    )


train_counts = patient_counts(
    train_patients
)

normal_counts = patient_counts(
    normal_patients
)

tumor_counts = patient_counts(
    tumor_patients
)


for name, counts in [
    ("TRAIN", train_counts),
    ("TEST NORMAL", normal_counts),
    ("TEST TUMOR", tumor_counts)
]:

    print(f"\n{name}")

    print(
        f"  Min: {counts.min()}"
    )

    print(
        f"  Max: {counts.max()}"
    )

    print(
        f"  Mean: {counts.mean():.2f}"
    )

    print(
        f"  Median: {np.median(counts):.2f}"
    )

    print(
        f"  Std: {counts.std():.2f}"
    )


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.hist(
    train_counts,
    bins=30,
    alpha=0.6,
    label="Train"
)

ax.hist(
    normal_counts,
    bins=30,
    alpha=0.6,
    label="Test Normal"
)

ax.hist(
    tumor_counts,
    bins=30,
    alpha=0.6,
    label="Test Tumor"
)

ax.set_xlabel(
    "Numero di slice per paziente"
)

ax.set_ylabel(
    "Numero di pazienti"
)

ax.set_title(
    "Distribuzione delle slice per paziente"
)

ax.legend()

save_figure(
    fig,
    "02_slices_per_patient.png"
)


# ============================================================
# 5. STATISTICHE PIXEL
# ============================================================

section("5. STATISTICHE DELLE IMMAGINI")


def analyze_files(files, split):

    records = []

    for i, path in enumerate(files):

        image = load_image(path)

        stats = image_statistics(
            image
        )

        stats["filename"] = path.name

        stats["patient_id"] = (
            extract_patient_id(
                path.name
            )
        )

        stats["slice"] = (
            extract_slice_number(
                path.name
            )
        )

        stats["split"] = split

        records.append(stats)

        if (
            (i + 1) % 500 == 0
        ):

            print(
                f"{split}: "
                f"{i + 1}/{len(files)}"
            )

    return records


all_records = []

all_records.extend(
    analyze_files(
        train_files,
        "train"
    )
)

all_records.extend(
    analyze_files(
        normal_files,
        "normal"
    )
)

all_records.extend(
    analyze_files(
        tumor_files,
        "tumor"
    )
)


image_df = pd.DataFrame(
    all_records
)


image_csv = (
    OUTPUT_DIR /
    "image_statistics.csv"
)

image_df.to_csv(
    image_csv,
    index=False
)


print(
    f"\nSalvato: {image_csv}"
)


# ============================================================
# 6. STATISTICHE RIASSUNTIVE
# ============================================================

section(
    "6. STATISTICHE INTENSITA' NORMAL / TUMOR"
)


for split in [
    "normal",
    "tumor"
]:

    subset = image_df[
        image_df["split"] == split
    ]

    print(
        f"\n{split.upper()}"
    )

    for column in [
        "mean",
        "std",
        "median",
        "p25",
        "p75",
        "p95",
        "nonzero_fraction"
    ]:

        print(
            f"{column:20s}: "
            f"mean={subset[column].mean():.4f} | "
            f"median={subset[column].median():.4f}"
        )



# ============================================================
# 7. DISTRIBUZIONE INTENSITA'
# ============================================================
section(
    "7. DISTRIBUZIONE DELLE INTENSITA'"
)

normal_sample, tumor_sample = (
    balanced_sample(
        normal_files,
        tumor_files,
        N_WASSERSTEIN
    )
)

normal_images = get_image_array(
    normal_sample
)

tumor_images = get_image_array(
    tumor_sample
)

normal_pixels = normal_images.flatten()
tumor_pixels = tumor_images.flatten()

# ------------------------------------------------------------
# Intensità globali
# ------------------------------------------------------------
wasserstein_global = (
    wasserstein_distance(
        normal_pixels,
        tumor_pixels
    )
)

print(
    f"Wasserstein globale: "
    f"{wasserstein_global:.6f}"
)

# ------------------------------------------------------------
# Intensità non-zero
# ------------------------------------------------------------
normal_nonzero = (
    normal_pixels[
        normal_pixels > 0
    ]
)

tumor_nonzero = (
    tumor_pixels[
        tumor_pixels > 0
    ]
)

wasserstein_nonzero = (
    wasserstein_distance(
        normal_nonzero,
        tumor_nonzero
    )
)

print(
    f"Wasserstein non-zero: "
    f"{wasserstein_nonzero:.6f}"
)

# ------------------------------------------------------------
# Istogramma separato Normal / Tumor
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharey=True
)

# NORMAL
axes[0].hist(
    normal_nonzero,
    bins=80,
    density=True,
    alpha=0.75,
    edgecolor="black",
    linewidth=0.5
)

axes[0].set_xlabel(
    "Intensità pixel non-zero"
)

axes[0].set_ylabel(
    "Densità"
)

axes[0].set_title(
    "Distribuzione intensità - Normal"
)

axes[0].grid(
    alpha=0.3,
    axis="y"
)

# TUMOR
axes[1].hist(
    tumor_nonzero,
    bins=80,
    density=True,
    alpha=0.75,
    edgecolor="black",
    linewidth=0.5
)

axes[1].set_xlabel(
    "Intensità pixel non-zero"
)

axes[1].set_title(
    "Distribuzione intensità - Tumor"
)

axes[1].grid(
    alpha=0.3,
    axis="y"
)

fig.suptitle(
    f"Distribuzione delle intensità\n"
    f"Wasserstein non-zero = {wasserstein_nonzero:.4f}",
    fontsize=13
)

plt.tight_layout()

save_figure(
    fig,
    "03_intensity_distribution.png"
)


# ============================================================
# 8. IMMAGINI MEDIE
# ============================================================

section(
    "8. ANALISI ANATOMICA"
)


mean_normal = normal_images.mean(
    axis=0
)

mean_tumor = tumor_images.mean(
    axis=0
)

difference = (
    mean_tumor -
    mean_normal
)


fig, ax = plt.subplots(
    1,
    3,
    figsize=(14, 5)
)


ax[0].imshow(
    mean_normal,
    cmap="gray"
)

ax[0].set_title(
    "Immagine media - Normal"
)

ax[0].axis("off")


ax[1].imshow(
    mean_tumor,
    cmap="gray"
)

ax[1].set_title(
    "Immagine media - Tumor"
)

ax[1].axis("off")


vmax = np.max(
    np.abs(difference)
)


ax[2].imshow(
    difference,
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax
)

ax[2].set_title(
    "Differenza Tumor - Normal"
)

ax[2].axis("off")


plt.tight_layout()

save_figure(
    fig,
    "04_mean_images.png"
)


# ============================================================
# 9. TUMOR FREQUENCY MAP
# ============================================================

section(
    "9. FREQUENZA SPAZIALE DEL TUMORE"
)


tumor_mask_files = []

for tumor_file in tumor_files:

    expected_name = (
        tumor_file.name.replace(
            "_flair_",
            "_seg_"
        )
    )

    mask_path = (
        ANNOTATION_DIR /
        expected_name
    )

    if mask_path.exists():

        tumor_mask_files.append(
            mask_path
        )


print(
    f"Maschere associate a immagini tumor: "
    f"{len(tumor_mask_files)}"
)


masks = []

for path in tumor_mask_files:

    mask = load_image(path)

    masks.append(
        mask > 0
    )


masks = np.asarray(
    masks,
    dtype=np.float32
)


frequency = masks.mean(
    axis=0
)


frequency_smooth = gaussian_filter(
    frequency,
    sigma=1
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


im = ax.imshow(
    frequency_smooth,
    cmap="hot"
)


plt.colorbar(
    im,
    ax=ax,
    label="Frequenza"
)

ax.set_title(
    "Frequenza spaziale delle lesioni"
)

ax.axis("off")


save_figure(
    fig,
    "05_tumor_frequency.png"
)



# ============================================================
# 10. TUMOR AREA
# ============================================================
section(
    "10. AREA DELLA LESIONE"
)

tumor_areas = []

for path in tumor_mask_files:

    mask = load_image(path)

    area = np.sum(
        mask > 0
    )

    percentage = (
        area /
        mask.size *
        100
    )

    tumor_areas.append(
        percentage
    )

tumor_areas = np.asarray(
    tumor_areas
)

print(
    f"Numero slice con tumore: "
    f"{len(tumor_areas)}"
)

print(
    f"Min: {tumor_areas.min():.4f}%"
)

print(
    f"Max: {tumor_areas.max():.4f}%"
)

print(
    f"Media: {tumor_areas.mean():.4f}%"
)

print(
    f"Mediana: {np.median(tumor_areas):.4f}%"
)

print(
    f"Std: {tumor_areas.std():.4f}%"
)

# ------------------------------------------------------------
# Grafici separati
# ------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)

# ISTOGRAMMA
axes[0].hist(
    tumor_areas,
    bins=40,
    alpha=0.75,
    edgecolor="black",
    linewidth=0.7
)

axes[0].axvline(
    tumor_areas.mean(),
    linestyle="--",
    linewidth=2,
    label=f"Media = {tumor_areas.mean():.2f}%"
)

axes[0].axvline(
    np.median(tumor_areas),
    linestyle=":",
    linewidth=2,
    label=f"Mediana = {np.median(tumor_areas):.2f}%"
)

axes[0].set_xlabel(
    "Area tumorale (%)"
)

axes[0].set_ylabel(
    "Frequenza"
)

axes[0].set_title(
    "Distribuzione dell'area tumorale"
)

axes[0].legend()

axes[0].grid(
    alpha=0.3,
    axis="y"
)

# BOXPLOT
axes[1].boxplot(
    tumor_areas,
    vert=True,
    widths=0.5
)

axes[1].set_ylabel(
    "Area tumorale (%)"
)

axes[1].set_title(
    "Boxplot dell'area tumorale"
)

axes[1].set_xticks([1])
axes[1].set_xticklabels(
    ["Tumor"]
)

axes[1].grid(
    alpha=0.3,
    axis="y"
)

fig.suptitle(
    "Distribuzione dell'area delle lesioni tumorali",
    fontsize=13
)

plt.tight_layout()

save_figure(
    fig,
    "06_tumor_area.png"
)

# ============================================================
# 11. ESEMPI NORMAL
# ============================================================

section(
    "11. ESEMPI VISIVI NORMAL"
)


n = min(
    N_EXAMPLES,
    len(normal_files)
)

selected = rng.choice(
    normal_files,
    size=n,
    replace=False
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()


for ax, path in zip(
    axes,
    selected
):

    image = load_image(
        path
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        path.stem,
        fontsize=8
    )

    ax.axis("off")


for ax in axes[n:]:

    ax.axis("off")


fig.suptitle(
    "Esempi di slice normali"
)

plt.tight_layout()

save_figure(
    fig,
    "07_normal_examples.png"
)


# ============================================================
# 12. ESEMPI TUMOR
# ============================================================

section(
    "12. ESEMPI VISIVI TUMOR"
)


n = min(
    N_EXAMPLES,
    len(tumor_files)
)

selected = rng.choice(
    tumor_files,
    size=n,
    replace=False
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()


for ax, path in zip(
    axes,
    selected
):

    image = load_image(
        path
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        path.stem,
        fontsize=8
    )

    ax.axis("off")


for ax in axes[n:]:

    ax.axis("off")


fig.suptitle(
    "Esempi di slice tumorali"
)

plt.tight_layout()

save_figure(
    fig,
    "08_tumor_examples.png"
)


# ============================================================
# 13. OVERLAY
# ============================================================

section(
    "13. VISUALIZZAZIONE FLAIR + SEGMENTATION"
)


available_pairs = []

for tumor_file in tumor_files:

    mask_name = (
        tumor_file.name.replace(
            "_flair_",
            "_seg_"
        )
    )

    mask_path = (
        ANNOTATION_DIR /
        mask_name
    )

    if mask_path.exists():

        available_pairs.append(
            (
                tumor_file,
                mask_path
            )
        )


n = min(
    N_EXAMPLES,
    len(available_pairs)
)

selected = rng.choice(
    len(available_pairs),
    size=n,
    replace=False
)


fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

axes = axes.flatten()


for ax, index in zip(
    axes,
    selected
):

    image_path, mask_path = (
        available_pairs[index]
    )

    image = load_image(
        image_path
    )

    mask = load_image(
        mask_path
    )


    ax.imshow(
        image,
        cmap="gray"
    )


    masked = np.ma.masked_where(
        mask == 0,
        mask
    )


    ax.imshow(
        masked,
        cmap="Reds",
        alpha=0.5
    )


    ax.set_title(
        image_path.stem,
        fontsize=8
    )

    ax.axis("off")


for ax in axes[n:]:

    ax.axis("off")


fig.suptitle(
    "Slice FLAIR con sovrapposizione della segmentation"
)

plt.tight_layout()

save_figure(
    fig,
    "09_overlay.png"
)


# ============================================================
# 14. PCA
# ============================================================

section(
    "14. PCA ESPLORATIVA"
)


# Campionamento bilanciato
n_normal = min(
    N_PCA // 2,
    len(normal_files)
)

n_tumor = min(
    N_PCA // 2,
    len(tumor_files)
)


pca_normal_files = rng.choice(
    normal_files,
    size=n_normal,
    replace=False
)

pca_tumor_files = rng.choice(
    tumor_files,
    size=n_tumor,
    replace=False
)


pca_files = list(
    pca_normal_files
) + list(
    pca_tumor_files
)


print(
    f"Immagini utilizzate per PCA: "
    f"{len(pca_files)}"
)


X = get_image_array(
    pca_files
)


X = X.reshape(
    len(X),
    -1
)


y = np.array(
    [0] * n_normal +
    [1] * n_tumor
)


print(
    f"Feature originali per immagine: "
    f"{X.shape[1]}"
)


pca = PCA(
    n_components=2,
    random_state=RANDOM_SEED
)


X_pca = pca.fit_transform(
    X
)


print(
    f"Varianza spiegata PC1: "
    f"{pca.explained_variance_ratio_[0]:.4f}"
)

print(
    f"Varianza spiegata PC2: "
    f"{pca.explained_variance_ratio_[1]:.4f}"
)

print(
    f"Varianza spiegata cumulativa: "
    f"{pca.explained_variance_ratio_.sum():.4f}"
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.scatter(
    X_pca[y == 0, 0],
    X_pca[y == 0, 1],
    s=8,
    alpha=0.5,
    label="Normal"
)


ax.scatter(
    X_pca[y == 1, 0],
    X_pca[y == 1, 1],
    s=8,
    alpha=0.5,
    label="Tumor"
)


ax.set_xlabel(
    "PC1"
)

ax.set_ylabel(
    "PC2"
)

ax.set_title(
    "PCA esplorativa"
)

ax.legend()


save_figure(
    fig,
    "10_pca.png"
)


# ============================================================
# 15. DATASET SUMMARY
# ============================================================

section(
    "15. DATASET SUMMARY"
)


summary_df = pd.DataFrame({

    "split": [
        "train",
        "test_normal",
        "test_tumor"
    ],

    "images": [
        len(train_files),
        len(normal_files),
        len(tumor_files)
    ],

    "patients": [
        len(train_patients),
        len(normal_patients),
        len(tumor_patients)
    ]
})


summary_path = (
    OUTPUT_DIR /
    "dataset_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


print(
    f"Salvato: {summary_path}"
)


# ============================================================
# 16. PATIENT SUMMARY
# ============================================================

patient_rows = []


for split, dictionary in [
    ("train", train_patients),
    ("normal", normal_patients),
    ("tumor", tumor_patients)
]:

    for patient, files in dictionary.items():

        slices = [
            extract_slice_number(
                path.name
            )
            for path in files
        ]

        slices = [
            s
            for s in slices
            if s is not None
        ]

        patient_rows.append({

            "patient_id": patient,

            "split": split,

            "num_slices": len(files),

            "min_slice": (
                min(slices)
                if slices
                else None
            ),

            "max_slice": (
                max(slices)
                if slices
                else None
            )
        })


patient_df = pd.DataFrame(
    patient_rows
)


patient_path = (
    OUTPUT_DIR /
    "patient_summary.csv"
)


patient_df.to_csv(
    patient_path,
    index=False
)


print(
    f"Salvato: {patient_path}"
)


# ============================================================
# 17. ANNOTATION SUMMARY
# ============================================================

section(
    "16. ANNOTATION SUMMARY"
)


annotation_rows = []


for mask_path in annotation_files:

    mask = load_image(
        mask_path
    )

    positive = (
        mask > 0
    )

    area = int(
        positive.sum()
    )

    annotation_rows.append({

        "filename":
            mask_path.name,

        "patient_id":
            extract_patient_id(
                mask_path.name
            ),

        "slice":
            extract_slice_number(
                mask_path.name
            ),

        "area_pixels":
            area,

        "area_fraction":
            area / mask.size,

        "unique_values":
            str(
                np.unique(mask).tolist()
            )
    })


annotation_df = pd.DataFrame(
    annotation_rows
)


annotation_path = (
    OUTPUT_DIR /
    "annotation_statistics.csv"
)


annotation_df.to_csv(
    annotation_path,
    index=False
)


print(
    f"Salvato: {annotation_path}"
)


# ============================================================
# 18. CONTROLLO DIMENSIONI
# ============================================================

section(
    "17. CONTROLLO DIMENSIONI"
)


size_counter = defaultdict(int)


for path in (
    train_files +
    normal_files +
    tumor_files +
    annotation_files
):

    with Image.open(path) as img:

        size_counter[
            img.size
        ] += 1


for size, count in sorted(
    size_counter.items()
):

    print(
        f"{size}: {count} immagini"
    )


# ============================================================
# 19. REPORT FINALE
# ============================================================

section(
    "18. EDA COMPLETATA"
)


print(
    "\nIl report non contiene "
    "conclusioni automatiche."
)

print(
    "I risultati devono essere "
    "interpretati a partire dalle "
    "statistiche e dalle figure."
)


print(
    f"\nReport:"
    f"\n{REPORT_FILE.resolve()}"
)

print(
    f"\nFigure:"
    f"\n{FIGURES_DIR.resolve()}"
)

print(
    "\nDataset originale NON modificato."
)


# ============================================================
# CHIUSURA
# ============================================================

report_file.flush()
report_file.close()

sys.stdout = sys.__stdout__
