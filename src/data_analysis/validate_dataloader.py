# ============================================================
# VALIDAZIONE DATALOADER E PREPROCESSING
# ============================================================

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from dataloader import get_dataset


# ============================================================
# CONFIGURAZIONE
# ============================================================

DATA_ROOT = "data"
IMG_SIZE = 64
BATCH_SIZE = 32

OUTPUT_DIR = Path("results") / "dataloader_validation"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# FUNZIONI UTILI
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def check_tensor(name, tensor):
    """Controlla proprietà fondamentali di un tensor."""

    print(f"\n{name}")

    print(f"  Shape:  {tuple(tensor.shape)}")
    print(f"  Dtype:  {tensor.dtype}")
    print(f"  Min:    {tensor.min().item():.6f}")
    print(f"  Max:    {tensor.max().item():.6f}")
    print(f"  Mean:   {tensor.mean().item():.6f}")
    print(f"  Std:    {tensor.std().item():.6f}")

    assert torch.isfinite(tensor).all(), (
        f"{name}: trovati NaN o Inf"
    )


def validate_image_tensor(
    tensor,
    expected_shape
):
    """Valida tensor immagine."""

    assert tuple(tensor.shape) == expected_shape, (
        f"Shape errata: {tuple(tensor.shape)} "
        f"attesa {expected_shape}"
    )

    assert tensor.dtype == torch.float32, (
        f"Dtype errato: {tensor.dtype}"
    )

    assert torch.isfinite(tensor).all(), (
        "Presenti NaN o Inf"
    )

    assert tensor.min() >= 0, (
        f"Valore minimo < 0: {tensor.min().item()}"
    )

    assert tensor.max() <= 1, (
        f"Valore massimo > 1: {tensor.max().item()}"
    )


def validate_mask_tensor(
    tensor,
    expected_shape
):
    """Valida tensor della segmentation mask."""

    assert tuple(tensor.shape) == expected_shape, (
        f"Shape mask errata: {tuple(tensor.shape)} "
        f"attesa {expected_shape}"
    )

    assert tensor.dtype == torch.float32, (
        f"Dtype mask errato: {tensor.dtype}"
    )

    assert torch.isfinite(tensor).all(), (
        "Mask contiene NaN o Inf"
    )

    unique_values = torch.unique(tensor)

    print(
        f"  Mask unique values: "
        f"{unique_values.tolist()}"
    )

    assert torch.all(
        (unique_values == 0) |
        (unique_values == 1)
    ), (
        "La mask non è binaria."
    )


# ============================================================
# 1. CARICAMENTO DATASET
# ============================================================

print_section(
    "1. CARICAMENTO DATASET"
)

print(
    f"Data root: {DATA_ROOT}"
)

print(
    f"Image size: {IMG_SIZE}x{IMG_SIZE}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)


train_ds = get_dataset(
    dataset_name="brats",
    data_root=DATA_ROOT,
    img_size=IMG_SIZE,
    mode="train"
)


test_ds = get_dataset(
    dataset_name="brats",
    data_root=DATA_ROOT,
    img_size=IMG_SIZE,
    mode="test"
)


print(
    f"\nTRAIN samples: {len(train_ds)}"
)

print(
    f"TEST samples:  {len(test_ds)}"
)


# ============================================================
# 2. DATASET STATISTICS
# ============================================================

print_section(
    "2. DATASET STATISTICS"
)


train_stats = train_ds.get_statistics()
test_stats = test_ds.get_statistics()


print("\nTRAIN")

for key, value in train_stats.items():
    print(
        f"  {key}: {value}"
    )


print("\nTEST")

for key, value in test_stats.items():
    print(
        f"  {key}: {value}"
    )


# ============================================================
# 3. CONTROLLO SINGOLO SAMPLE TRAIN
# ============================================================

print_section(
    "3. VALIDAZIONE SAMPLE TRAIN"
)


train_sample = train_ds[0]


print(
    f"Name: {train_sample['name']}"
)

print(
    f"Label: {train_sample['label']}"
)


check_tensor(
    "TRAIN image",
    train_sample["img"]
)


validate_image_tensor(
    train_sample["img"],
    (1, IMG_SIZE, IMG_SIZE)
)


assert train_sample["label"] == 0, (
    "Il training set deve contenere "
    "solamente immagini healthy."
)


# ============================================================
# 4. CONTROLLO SINGOLO SAMPLE TEST
# ============================================================

print_section(
    "4. VALIDAZIONE SAMPLE TEST"
)


test_sample = test_ds[0]


print(
    f"Name: {test_sample['name']}"
)

print(
    f"Label: {test_sample['label']}"
)


check_tensor(
    "TEST image",
    test_sample["img"]
)


validate_image_tensor(
    test_sample["img"],
    (1, IMG_SIZE, IMG_SIZE)
)


check_tensor(
    "TEST mask",
    test_sample["mask"]
)


validate_mask_tensor(
    test_sample["mask"],
    (1, IMG_SIZE, IMG_SIZE)
)


# ============================================================
# 5. DISTRIBUZIONE LABEL
# ============================================================

print_section(
    "5. LABEL DISTRIBUTION"
)


train_labels = np.asarray(
    train_ds.labels
)

test_labels = np.asarray(
    test_ds.labels
)


print(
    f"TRAIN label 0: "
    f"{np.sum(train_labels == 0)}"
)

print(
    f"TRAIN label 1: "
    f"{np.sum(train_labels == 1)}"
)


print(
    f"\nTEST label 0: "
    f"{np.sum(test_labels == 0)}"
)

print(
    f"TEST label 1: "
    f"{np.sum(test_labels == 1)}"
)


assert np.all(
    train_labels == 0
), (
    "Il TRAIN contiene label diverse da 0."
)


assert set(
    np.unique(test_labels)
).issubset({0, 1}), (
    "Il TEST contiene label non valide."
)


# ============================================================
# 6. CONTROLLO DATASET COMPLETO
# ============================================================

print_section(
    "6. VALIDAZIONE DATASET COMPLETO"
)


print(
    "Controllo immagini TRAIN..."
)


for i in range(len(train_ds)):

    sample = train_ds[i]

    validate_image_tensor(
        sample["img"],
        (1, IMG_SIZE, IMG_SIZE)
    )

    assert sample["label"] == 0


print(
    "TRAIN: OK"
)


print(
    "\nControllo immagini TEST..."
)


for i in range(len(test_ds)):

    sample = test_ds[i]

    validate_image_tensor(
        sample["img"],
        (1, IMG_SIZE, IMG_SIZE)
    )

    validate_mask_tensor(
        sample["mask"],
        (1, IMG_SIZE, IMG_SIZE)
    )


print(
    "TEST: OK"
)


# ============================================================
# 7. CREAZIONE DATALOADER
# ============================================================

print_section(
    "7. PYTORCH DATALOADER"
)


train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print(
    f"TRAIN batches: {len(train_loader)}"
)

print(
    f"TEST batches:  {len(test_loader)}"
)


# ============================================================
# 8. CONTROLLO BATCH TRAIN
# ============================================================

print_section(
    "8. VALIDAZIONE BATCH TRAIN"
)


train_batch = next(
    iter(train_loader)
)


print(
    f"Images: {train_batch['img'].shape}"
)

print(
    f"Labels: {train_batch['label'].shape}"
)


expected_train_shape = (
    min(BATCH_SIZE, len(train_ds)),
    1,
    IMG_SIZE,
    IMG_SIZE
)


assert tuple(
    train_batch["img"].shape
) == expected_train_shape


assert train_batch["label"].shape[0] == (
    expected_train_shape[0]
)


validate_image_tensor(
    train_batch["img"],
    expected_train_shape
)


assert torch.all(
    train_batch["label"] == 0
)


print(
    "TRAIN batch: OK"
)


# ============================================================
# 9. CONTROLLO BATCH TEST
# ============================================================

print_section(
    "9. VALIDAZIONE BATCH TEST"
)


test_batch = next(
    iter(test_loader)
)


print(
    f"Images: {test_batch['img'].shape}"
)

print(
    f"Labels: {test_batch['label'].shape}"
)

print(
    f"Masks:  {test_batch['mask'].shape}"
)


expected_test_shape = (
    min(BATCH_SIZE, len(test_ds)),
    1,
    IMG_SIZE,
    IMG_SIZE
)


assert tuple(
    test_batch["img"].shape
) == expected_test_shape


assert tuple(
    test_batch["mask"].shape
) == expected_test_shape


validate_image_tensor(
    test_batch["img"],
    expected_test_shape
)


validate_mask_tensor(
    test_batch["mask"],
    expected_test_shape
)


print(
    "TEST batch: OK"
)


# ============================================================
# 10. VISUALIZZAZIONE PREPROCESSING
# ============================================================

print_section(
    "10. VISUALIZZAZIONE PREPROCESSING"
)


n_examples = min(
    6,
    len(test_ds)
)


indices = np.linspace(
    0,
    len(test_ds) - 1,
    n_examples,
    dtype=int
)


fig, axes = plt.subplots(
    2,
    n_examples,
    figsize=(3 * n_examples, 6)
)


if n_examples == 1:
    axes = axes.reshape(2, 1)


for col, index in enumerate(indices):

    sample = test_ds[index]

    image = (
        sample["img"]
        .squeeze(0)
        .numpy()
    )

    mask = (
        sample["mask"]
        .squeeze(0)
        .numpy()
    )

    axes[0, col].imshow(
        image,
        cmap="gray"
    )

    axes[0, col].set_title(
        f"{sample['name']}\n"
        f"label={sample['label']}"
    )

    axes[0, col].axis("off")


    axes[1, col].imshow(
        image,
        cmap="gray"
    )

    masked = np.ma.masked_where(
        mask == 0,
        mask
    )

    axes[1, col].imshow(
        masked,
        cmap="Reds",
        alpha=0.5
    )

    axes[1, col].set_title(
        "Preprocessed + mask"
    )

    axes[1, col].axis("off")


fig.suptitle(
    "Validazione preprocessing DataLoader",
    fontsize=14
)

plt.tight_layout()


figure_path = (
    FIGURES_DIR /
    "01_preprocessing_validation.png"
)


fig.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print(
    f"Figura salvata: {figure_path}"
)


# ============================================================
# 11. STATISTICHE DELLE IMMAGINI DOPO PREPROCESSING
# ============================================================

print_section(
    "11. STATISTICHE DOPO PREPROCESSING"
)


def compute_dataset_tensor_statistics(
    dataset,
    max_samples=None
):

    if max_samples is None:
        max_samples = len(dataset)

    max_samples = min(
        max_samples,
        len(dataset)
    )

    means = []
    stds = []
    mins = []
    maxs = []

    for i in range(max_samples):

        image = dataset[i]["img"]

        means.append(
            image.mean().item()
        )

        stds.append(
            image.std().item()
        )

        mins.append(
            image.min().item()
        )

        maxs.append(
            image.max().item()
        )

    return {
        "samples": max_samples,
        "mean_of_means": float(
            np.mean(means)
        ),
        "mean_of_stds": float(
            np.mean(stds)
        ),
        "global_min": float(
            np.min(mins)
        ),
        "global_max": float(
            np.max(maxs)
        )
    }


train_processed_stats = (
    compute_dataset_tensor_statistics(
        train_ds
    )
)


test_processed_stats = (
    compute_dataset_tensor_statistics(
        test_ds
    )
)


print("\nTRAIN:")

for key, value in train_processed_stats.items():
    print(
        f"  {key}: {value}"
    )


print("\nTEST:")

for key, value in test_processed_stats.items():
    print(
        f"  {key}: {value}"
    )


# ============================================================
# 12. SALVATAGGIO RISULTATI
# ============================================================

print_section(
    "12. SALVATAGGIO RISULTATI"
)


results = {
    "configuration": {
        "img_size": IMG_SIZE,
        "channels": 1,
        "batch_size": BATCH_SIZE,
        "normalization": "[0,1]",
        "augmentation": None,
        "seed": SEED
    },

    "train": {
        "samples": len(train_ds),
        "statistics": train_processed_stats
    },

    "test": {
        "samples": len(test_ds),
        "statistics": test_processed_stats
    }
}


results_path = (
    OUTPUT_DIR /
    "dataloader_validation.json"
)


with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    f"Risultati salvati in:\n"
    f"{results_path}"
)


# ============================================================
# 13. CHECK FINALE
# ============================================================

print_section(
    "13. CHECK FINALE"
)


print("✓ Dataset TRAIN caricato")
print("✓ Dataset TEST caricato")
print("✓ Shape immagini corretta")
print("✓ Shape maschere corretta")
print("✓ Immagini grayscale")
print("✓ Tensor float32")
print("✓ Range immagini [0,1]")
print("✓ Maschere binarie")
print("✓ Label TRAIN = 0")
print("✓ Label TEST ∈ {0,1}")
print("✓ Nessun NaN/Inf")
print("✓ Batch PyTorch validati")
print("✓ Preprocessing visualizzato")


print()
print("=" * 70)
print("DATALOADER VALIDATION COMPLETATA CON SUCCESSO")
print("=" * 70)