import os
import time
from typing import Optional, List, Dict, Callable

import numpy as np
from PIL import Image
from joblib import Parallel, delayed

import torch
from torch.utils import data
from torchvision import transforms


# ==========================================================
# CONFIGURAZIONE
# ==========================================================

SEED: int = 42
DEFAULT_IMG_SIZE: int = 64


# ==========================================================
# RIPRODUCIBILITÀ
# ==========================================================

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# CARICAMENTO IMMAGINI
# ==========================================================

def load_single_image(
    file_name: str,
    img_dir: str,
    img_size: int,
    resample: int = Image.BILINEAR
) -> Image.Image:
    """
    Carica e ridimensiona una singola immagine grayscale.

    Args:
        file_name:
            Nome del file.
        img_dir:
            Directory contenente il file.
        img_size:
            Dimensione finale quadrata.
        resample:
            Metodo di interpolazione PIL.

    Returns:
        Immagine PIL grayscale ridimensionata.
    """

    path = os.path.join(img_dir, file_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Immagine non trovata: {path}"
        )

    image = Image.open(path).convert("L")

    image = image.resize(
        (img_size, img_size),
        resample=resample
    )

    return image


def parallel_load(
    img_dir: str,
    img_list: List[str],
    img_size: int,
    resample: str = "bilinear",
    verbose: int = 0
) -> List[Image.Image]:
    """
    Carica immagini in parallelo.

    Args:
        img_dir:
            Directory contenente le immagini.
        img_list:
            Lista dei file.
        img_size:
            Dimensione finale.
        resample:
            "bilinear" oppure "nearest".
        verbose:
            Verbosity di joblib.

    Returns:
        Lista di immagini PIL.

    Raises:
        ValueError:
            Se il metodo di interpolazione non è valido.
    """

    if resample == "bilinear":
        resample_method = Image.BILINEAR

    elif resample == "nearest":
        resample_method = Image.NEAREST

    else:
        raise ValueError(
            f"Metodo di interpolazione non valido: {resample}. "
            f"Usare 'bilinear' oppure 'nearest'."
        )

    images = Parallel(
        n_jobs=-1,
        verbose=verbose
    )(
        delayed(load_single_image)(
            file_name=file_name,
            img_dir=img_dir,
            img_size=img_size,
            resample=resample_method
        )
        for file_name in img_list
    )

    return images


# ==========================================================
# DATASET BRATS2021
# ==========================================================

class BraTSDataset(data.Dataset):
    """
    Dataset PyTorch per BraTS2021 nel formato MedIAnomaly.

    Struttura attesa:

        BraTS2021/
        ├── train/
        │   ├── image_001.png
        │   ├── image_002.png
        │   └── ...
        │
        └── test/
            ├── normal/
            │   ├── image_001.png
            │   └── ...
            │
            ├── tumor/
            │   ├── image_001.png
            │   └── ...
            │
            └── annotation/
                ├── image_001_seg.png
                └── ...

    TRAIN:
        label = 0 per tutte le immagini.

    TEST:
        label = 0 -> healthy
        label = 1 -> tumor

    Le immagini vengono convertite in:
        torch.Tensor [1, H, W]

    con valori nell'intervallo:
        [0, 1]
    """

    def __init__(
        self,
        main_path: str,
        img_size: int = DEFAULT_IMG_SIZE,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ) -> None:

        super().__init__()

        if mode not in ("train", "test"):
            raise ValueError(
                f"mode deve essere 'train' oppure 'test', ricevuto: {mode}"
            )

        if img_size <= 0:
            raise ValueError(
                f"img_size deve essere positivo, ricevuto: {img_size}"
            )

        self.mode = mode
        self.root = main_path
        self.res = img_size

        self.labels: List[int] = []
        self.img_ids: List[str] = []
        self.patient_ids: List[str] = []
        self.slices: List[Image.Image] = []
        self.masks: List[np.ndarray] = []
        

        self.transform = (
            transform
            if transform is not None
            else get_transforms()
        )

        print()
        print("=" * 70)
        print(f"CARICAMENTO BRATS2021 — {mode.upper()}")
        print("=" * 70)
        print(f"Root:       {main_path}")
        print(f"Resolution: {img_size}x{img_size}")
        print("Channels:   1 (grayscale)")
        print("Range:      [0, 1]")
        print()

        if mode == "train":
            self._load_train()

        else:
            self._load_test()

        self._validate_dataset()

    # ======================================================
    # TRAIN
    # ======================================================

    def _load_train(self) -> None:
        """
        Carica esclusivamente le immagini sane del training set.
        """

        train_dir = os.path.join(
            self.root,
            "train"
        )

        if not os.path.isdir(train_dir):
            raise FileNotFoundError(
                f"Directory TRAIN non trovata:\n{train_dir}"
            )

        train_imgs = sorted(
            self._get_image_files(train_dir)
        )

        if len(train_imgs) == 0:
            raise FileNotFoundError(
                f"Nessuna immagine trovata in:\n{train_dir}"
            )

        print(
            f"[TRAIN] Trovate {len(train_imgs)} immagini sane."
        )

        t0 = time.time()

        self.slices = parallel_load(
            img_dir=train_dir,
            img_list=train_imgs,
            img_size=self.res,
            resample="bilinear"
        )

        self.labels = [0] * len(train_imgs)

        self.img_ids = [
            os.path.splitext(file)[0]
            for file in train_imgs
        ]

        self.patient_ids = [
            os.path.splitext(file)[0].split("_flair_")[0]
            for file in train_imgs
        ]

        elapsed = time.time() - t0

        print(
            f"[TRAIN] Caricamento completato in "
            f"{elapsed:.2f}s"
        )

    # ======================================================
    # TEST
    # ======================================================

    def _load_test(self) -> None:
        """
        Carica immagini sane, tumorali e relative maschere.
        """

        normal_dir = os.path.join(
            self.root,
            "test",
            "normal"
        )

        tumor_dir = os.path.join(
            self.root,
            "test",
            "tumor"
        )

        annotation_dir = os.path.join(
            self.root,
            "test",
            "annotation"
        )

        for directory in (
            normal_dir,
            tumor_dir,
            annotation_dir
        ):
            if not os.path.isdir(directory):
                raise FileNotFoundError(
                    f"Directory non trovata:\n{directory}"
                )

        normal_imgs = sorted(
            self._get_image_files(normal_dir)
        )

        tumor_imgs = sorted(
            self._get_image_files(tumor_dir)
        )

        if len(normal_imgs) == 0:
            raise FileNotFoundError(
                "Nessuna immagine sana trovata in:\n"
                f"{normal_dir}"
            )

        if len(tumor_imgs) == 0:
            raise FileNotFoundError(
                "Nessuna immagine tumorale trovata in:\n"
                f"{tumor_dir}"
            )

        print(
            f"[TEST] Healthy images: {len(normal_imgs)}"
        )

        print(
            f"[TEST] Tumor images:   {len(tumor_imgs)}"
        )

        # --------------------------------------------------
        # MAPPING MASCHERE
        # --------------------------------------------------

        tumor_masks = [
            self._find_mask_file(
                image_name,
                annotation_dir
            )
            for image_name in tumor_imgs
        ]

        t0 = time.time()

        # --------------------------------------------------
        # IMMAGINI HEALTHY
        # --------------------------------------------------

        normal_slices = parallel_load(
            img_dir=normal_dir,
            img_list=normal_imgs,
            img_size=self.res,
            resample="bilinear"
        )

        # --------------------------------------------------
        # IMMAGINI TUMOR
        # --------------------------------------------------

        tumor_slices = parallel_load(
            img_dir=tumor_dir,
            img_list=tumor_imgs,
            img_size=self.res,
            resample="bilinear"
        )

        self.slices = (
            normal_slices +
            tumor_slices
        )

        # --------------------------------------------------
        # LABELS
        # --------------------------------------------------

        self.labels = (
            [0] * len(normal_imgs) +
            [1] * len(tumor_imgs)
        )

        # --------------------------------------------------
        # IDS
        # --------------------------------------------------

        self.img_ids = (
            [
                os.path.splitext(file)[0]
                for file in normal_imgs
            ]
            +
            [
                os.path.splitext(file)[0]
                for file in tumor_imgs
            ]
        )

        self.patient_ids = (
            [
                os.path.splitext(file)[0].split("_flair_")[0]
                for file in normal_imgs
            ]
            +
            [
                os.path.splitext(file)[0].split("_flair_")[0]
                for file in tumor_imgs
            ]
        )

        # --------------------------------------------------
        # MASCHERE HEALTHY
        # --------------------------------------------------

        normal_masks = [
            np.zeros(
                (self.res, self.res),
                dtype=np.float32
            )
            for _ in normal_imgs
        ]

        # --------------------------------------------------
        # MASCHERE TUMOR
        # --------------------------------------------------

        tumor_mask_images = parallel_load(
            img_dir=annotation_dir,
            img_list=tumor_masks,
            img_size=self.res,
            resample="nearest"
        )

        tumor_masks_np = []

        for mask in tumor_mask_images:

            mask_np = np.asarray(
                mask,
                dtype=np.float32
            )

            mask_np = (
                mask_np > 0
            ).astype(np.float32)

            tumor_masks_np.append(
                mask_np
            )

        self.masks = (
            normal_masks +
            tumor_masks_np
        )

        elapsed = time.time() - t0

        print(
            f"[TEST] Caricamento completato in "
            f"{elapsed:.2f}s"
        )

    # ======================================================
    # FILE UTILITY
    # ======================================================

    @staticmethod
    def _get_image_files(
        directory: str
    ) -> List[str]:
        """
        Restituisce esclusivamente file immagine validi.
        """

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        )

        files = []

        for file_name in os.listdir(directory):

            if file_name.lower().endswith(
                valid_extensions
            ):
                files.append(file_name)

        return sorted(files)

    @staticmethod
    def _find_mask_file(
        image_name: str,
        annotation_dir: str
    ) -> str:
        """
        Trova la maschera corrispondente all'immagine tumorale.

        Supporta diversi naming convention.
        """

        stem, _ = os.path.splitext(image_name)

        candidates = [
            f"{stem}_seg.png",
            f"{stem}_seg.jpg",
            f"{stem}_seg.jpeg",
            f"{stem}_seg.tif",
        ]

        # Caso comune:
        # image_flair.png -> image_seg.png
        if "flair" in stem.lower():

            seg_stem = stem.lower().replace(
                "flair",
                "seg"
            )

            candidates.extend([
                f"{seg_stem}.png",
                f"{seg_stem}.jpg",
                f"{seg_stem}.jpeg",
                f"{seg_stem}.tif",
            ])

        for candidate in candidates:

            path = os.path.join(
                annotation_dir,
                candidate
            )

            if os.path.isfile(path):
                return candidate

        raise FileNotFoundError(
            "\nMaschera non trovata per:"
            f"\n  {image_name}"
            f"\n\nDirectory annotation:"
            f"\n  {annotation_dir}"
            f"\n\nCandidati cercati:"
            f"\n  {candidates}"
        )

    # ======================================================
    # VALIDATION
    # ======================================================

    def _validate_dataset(self) -> None:
        """
        Controlla la coerenza interna del dataset.
        """

        n = len(self.slices)

        if len(self.labels) != n:
            raise RuntimeError(
                "Numero immagini e labels non coincide."
            )

        if len(self.img_ids) != n:
            raise RuntimeError(
                "Numero immagini e img_ids non coincide."
            )

        if len(self.patient_ids) != n:
            raise RuntimeError(
                "Numero immagini e patient_ids non coincide."
            )

        if self.mode == "test":

            if len(self.masks) != n:
                raise RuntimeError(
                    "Numero immagini e maschere non coincide."
                )

        if n == 0:
            raise RuntimeError(
                "Dataset vuoto."
            )

        print()
        print("Dataset validation: OK")
        print(f"  Samples: {n}")
        print()

    # ======================================================
    # GET ITEM
    # ======================================================

    def __getitem__(
        self,
        index: int
    ) -> Dict:
        """
        Restituisce un campione.

        TRAIN:
            img
            label
            name
            patient_id

        TEST:
            img
            label
            name
            mask
            patient_id
        """

        img = self.slices[index]

        img = self.transform(img)

        label = int(
            self.labels[index]
        )

        name = self.img_ids[index]

        if self.mode == "train":

            return {
                "img": img,
                "label": label,
                "name": name,
                "patient_id": self.patient_ids[index]
            }

        # TEST

        mask = torch.from_numpy(
            self.masks[index]
        ).float().unsqueeze(0)

        return {
            "img": img,
            "label": label,
            "name": name,
            "patient_id": self.patient_ids[index],
            "mask": mask
        }

    # ======================================================
    # LEN
    # ======================================================

    def __len__(self) -> int:
        """
        Numero totale di campioni.
        """

        return len(self.slices)

    # ======================================================
    # STATISTICS
    # ======================================================

    def get_statistics(self) -> Dict:
        """
        Calcola statistiche del dataset.
        """

        labels = np.asarray(
            self.labels
        )

        n_samples = len(labels)

        n_healthy = int(
            np.sum(labels == 0)
        )

        n_anomalous = int(
            np.sum(labels == 1)
        )

        anomaly_rate = (
            n_anomalous / n_samples
            if n_samples > 0
            else 0.0
        )

        return {
            "n_samples": n_samples,
            "n_healthy": n_healthy,
            "n_anomalous": n_anomalous,
            "anomaly_rate": float(
                anomaly_rate
            ),
            "img_shape": (
                self.res,
                self.res
            ),
            "channels": 1,
            "value_range": "[0, 1]"
        }


# ==========================================================
# TRANSFORMS
# ==========================================================

def get_transforms(
    is_grayscale: bool = True
) -> transforms.Compose:
    """
    Trasformazioni definitive per la baseline.

    Pipeline:

        PIL image
            ↓
        ToTensor()
            ↓
        [0, 1]

    NON viene utilizzato Normalize(mean=0.5, std=0.5)
    perché vogliamo mantenere i valori nell'intervallo
    naturale [0,1].

    La standardizzazione delle feature viene effettuata
    successivamente nello script Isolation Forest.
    """

    if not is_grayscale:
        raise ValueError(
            "La baseline BraTS utilizza immagini grayscale."
        )

    return transforms.Compose([
        transforms.ToTensor()
    ])


# ==========================================================
# FACTORY
# ==========================================================

def get_dataset(
    dataset_name: str,
    img_size: int = DEFAULT_IMG_SIZE,
    mode: str = "train"
) -> BraTSDataset:
    """
    Factory per creare il dataset.

    Args:
        dataset_name:
            Deve essere "brats".
        data_root:
            Directory contenente BraTS2021.
        img_size:
            Risoluzione finale.
        mode:
            "train" oppure "test".

    Returns:
        BraTSDataset.
    """

    if dataset_name.lower() != "brats":
        raise ValueError(
            f"Dataset non supportato: {dataset_name}. "
            f"Utilizzare 'brats'."
        )

    transform = get_transforms(
        is_grayscale=True
    )

    path = os.path.join(
        "BraTS2021"
    )

    return BraTSDataset(
        main_path=path,
        img_size=img_size,
        transform=transform,
        mode=mode
    )


# ==========================================================
# DATASET SUMMARY
# ==========================================================

def print_dataset_summary(
    dataset: BraTSDataset,
    name: str
) -> None:
    """
    Stampa un riepilogo leggibile del dataset.
    """

    stats = dataset.get_statistics()

    print("=" * 60)
    print(f"{name}")
    print("=" * 60)

    print(
        f"Samples:        {stats['n_samples']}"
    )

    print(
        f"Healthy:        {stats['n_healthy']}"
    )

    print(
        f"Anomalous:      {stats['n_anomalous']}"
    )

    print(
        f"Anomaly rate:   {stats['anomaly_rate']:.2%}"
    )

    print(
        f"Image shape:    {stats['img_shape']}"
    )

    print(
        f"Channels:       {stats['channels']}"
    )

    print(
        f"Value range:    {stats['value_range']}"
    )

    print("=" * 60)
    print()


# ==========================================================
# INTEGRITY TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("BRATS2021 DATALOADER — INTEGRITY TEST")
    print("=" * 70)

    try:

        # ==================================================
        # TRAIN
        # ==================================================

        print()
        print("[1/2] Loading TRAIN...")

        train_ds = get_dataset(
            dataset_name="brats",
            img_size=64,
            mode="train"
        )

        print("\nFirst TRAIN samples:")
        for i in range(10):
            sample = train_ds[i]
            print(
                f"  {i}: "
                f"{sample['name']} -> "
                f"{sample['patient_id']}"
            )


        print_dataset_summary(
            train_ds,
            "TRAIN SET"
        )

        sample_train = train_ds[0]

        print(
            f"TRAIN patient_id: "
            f"{sample_train['patient_id']}"
        )    

        print(
            f"TRAIN image shape: "
            f"{sample_train['img'].shape}"
        )

        print(
            f"TRAIN image dtype: "
            f"{sample_train['img'].dtype}"
        )

        print(
            f"TRAIN min value: "
            f"{sample_train['img'].min().item():.4f}"
        )

        print(
            f"TRAIN max value: "
            f"{sample_train['img'].max().item():.4f}"
        )

        # ==================================================
        # TEST
        # ==================================================

        print()
        print("[2/2] Loading TEST...")

        test_ds = get_dataset(
            dataset_name="brats",
            img_size=64,
            mode="test"
        )

        #temporaneo
        print("\n" + "=" * 70)
        print("CHECK PATIENT IDS")
        print("=" * 70)

        print("\nPrime 30 immagini TEST:")

        for i in range(min(30, len(test_ds))):

            print(
                f"{i:3d} | "
                f"image={test_ds.img_ids[i]:40s} | "
                f"patient={test_ds.patient_ids[i]:20s} | "
                f"label={test_ds.labels[i]}"
            )

        print("\nNumero pazienti unici:")

        patient_ids = np.asarray(test_ds.patient_ids)
        labels = np.asarray(test_ds.labels)

        print(
            "Totale:",
            len(np.unique(patient_ids))
        )

        print(
            "Healthy:",
            len(np.unique(patient_ids[labels == 0]))
        )

        print(
            "Tumor:",
            len(np.unique(patient_ids[labels == 1]))
        )




#fine test
        print_dataset_summary(
            test_ds,
            "TEST SET"
        )

        sample_test = test_ds[0]


        print(
            f"TEST patient_id: "
            f"{sample_test['patient_id']}"
        )

        print(
            f"TEST image shape: "
            f"{sample_test['img'].shape}"
        )

        print(
            f"TEST mask shape: "
            f"{sample_test['mask'].shape}"
        )

        print(
            f"TEST image dtype: "
            f"{sample_test['img'].dtype}"
        )

        print(
            f"TEST image min: "
            f"{sample_test['img'].min().item():.4f}"
        )

        print(
            f"TEST image max: "
            f"{sample_test['img'].max().item():.4f}"
        )

        # ==================================================
        # DATALOADER
        # ==================================================

        from torch.utils.data import DataLoader

        train_loader = DataLoader(
            train_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        train_batch = next(
            iter(train_loader)
        )

        test_batch = next(
            iter(test_loader)
        )

        print()
        print("TRAIN batch:")
        print(
            f"  img:   {train_batch['img'].shape}"
        )
        print(
            f"  label: {train_batch['label'].shape}"
        )

        print()
        print("TEST batch:")
        print(
            f"  img:   {test_batch['img'].shape}"
        )
        print(
            f"  label: {test_batch['label'].shape}"
        )
        print(
            f"  mask:  {test_batch['mask'].shape}"
        )

        # ==================================================
        # FINAL CHECK
        # ==================================================

        assert (
            train_batch["img"].shape[1:]
            == (1, 64, 64)
        )

        assert (
            test_batch["img"].shape[1:]
            == (1, 64, 64)
        )

        assert (
            test_batch["mask"].shape[1:]
            == (1, 64, 64)
        )

        assert (
            train_batch["img"].min() >= 0
        )

        assert (
            train_batch["img"].max() <= 1
        )

        assert (
            test_batch["img"].min() >= 0
        )

        assert (
            test_batch["img"].max() <= 1
        )

        print()
        print("=" * 70)
        print("DATALOADER TEST COMPLETATO CON SUCCESSO")
        print("=" * 70)
        print()
        print("Configurazione:")
        print("  Resolution:     64 x 64")
        print("  Channels:       1")
        print("  Normalization:  [0, 1]")
        print("  Augmentation:   None")
        print("  Train:          Healthy only")
        print("  Test:           Healthy + Tumor")
        print("  Seed:            42")
        print("=" * 70)

    except Exception as e:

        print()
        print("=" * 70)
        print("ERRORE DURANTE IL TEST DEL DATALOADER")
        print("=" * 70)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        