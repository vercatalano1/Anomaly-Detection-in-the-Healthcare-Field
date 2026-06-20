"""
DATALOADER DEFINITIVO per BraTS2021 (MedIAnomaly)

Caratteristiche:
✓ Type hints completi (Python 3.9+)
✓ Docstring per ogni funzione
✓ Seed riproducibilità
✓ Robust error handling
✓ Stats method per dataset analysis
✓ Production-ready
"""

import os
import time
from typing import Tuple, Optional, List, Dict
import numpy as np
from PIL import Image
from joblib import Parallel, delayed

import torch
from torch.utils import data
from torchvision import transforms


# ==========================================================
# RIPRODUCIBILITÀ
# ==========================================================

SEED: int = 42

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
    mode: str,
    img_size: int,
    resample: int
) -> Image.Image:
    """
    Carica una singola immagine e la preprocessing.
    
    Args:
        file_name: nome file da caricare
        img_dir: directory contenente il file
        mode: modalità PIL ("L" per grayscale, "RGB" per colore)
        img_size: dimensione target (quadrata)
        resample: metodo interpolazione PIL
    
    Returns:
        Image.Image preprocessata
    """
    return (
        Image.open(os.path.join(img_dir, file_name))
        .convert(mode)
        .resize((img_size, img_size), resample=resample)
    )


def parallel_load(
    img_dir: str,
    img_list: List[str],
    img_size: int,
    n_channel: int = 1,
    resample: str = "bilinear",
    verbose: int = 0
) -> List[Image.Image]:
    """
    Caricamento parallelo delle immagini su multicore CPU.
    
    Args:
        img_dir: directory base
        img_list: lista filename da caricare
        img_size: dimensione target
        n_channel: numero canali (1=grayscale, 3=RGB)
        resample: "bilinear" o "nearest"
        verbose: verbosity della joblib.Parallel
    
    Returns:
        Lista di Image.Image preprocessate
    
    Raises:
        ValueError: se resample non è valido
    """
    mode = "L" if n_channel == 1 else "RGB"

    if resample == "bilinear":
        resample_method = Image.BILINEAR
    elif resample == "nearest":
        resample_method = Image.NEAREST
    else:
        raise ValueError(f"Metodo di interpolazione '{resample}' non valido. Usa 'bilinear' o 'nearest'.")

    images = Parallel(n_jobs=-1, verbose=verbose)(
        delayed(load_single_image)(
            file_name,
            img_dir,
            mode,
            img_size,
            resample_method
        )
        for file_name in img_list
    )
    
    return images


# ==========================================================
# DATASET BraTS2021
# ==========================================================

class BraTSDataset(data.Dataset):
    """
    PyTorch Dataset per BraTS2021 (MedIAnomaly format).
    
    Supporta:
    - Train mode: carica solo immagini sane da train/
    - Test mode: carica sane da test/normal/ + anomale da test/tumor/ + maschere
    - Context encoding: masking casuale per self-supervised learning
    
    Attributes:
        mode (str): "train" o "test"
        root (str): percorso base BraTS2021
        res (int): risoluzione immagini
        labels (List[int]): 0=sano, 1=anomalo
        img_ids (List[str]): ID immagini
        slices (List[Image.Image]): immagini caricate
        masks (List[np.ndarray]): maschere segmentazione (solo test)
    """
    
    def __init__(
        self,
        main_path: str,
        img_size: int = 64,
        transform: Optional[callable] = None,
        mode: str = "train",
        context_encoding: bool = False
    ) -> None:
        """
        Inizializza il dataset.
        
        Args:
            main_path: percorso a BraTS2021/
            img_size: dimensione target immagini
            transform: torchvision.transforms.Compose
            mode: "train" o "test"
            context_encoding: abilita masking casuale per SSL
        
        Raises:
            AssertionError: se mode non è "train" o "test"
            FileNotFoundError: se le directory richieste non esistono
        """
        super().__init__()
        
        assert mode in ["train", "test"], f"mode deve essere 'train' o 'test', got {mode}"

        self.mode: str = mode
        self.root: str = main_path
        self.res: int = img_size

        self.labels: List[int] = []
        self.masks: List[np.ndarray] = []
        self.img_ids: List[str] = []
        self.slices: List[Image.Image] = []

        self.transform: callable = transform if transform is not None else lambda x: x

        # Context Encoding per SSL
        if context_encoding:
            self.random_mask = transforms.RandomErasing(
                p=1.0,
                scale=(0.02, 0.08),
                ratio=(0.5, 2.0),
                value=-1
            )
        else:
            self.random_mask = None

        print(f"\n[{mode.upper()}] Caricamento BraTS2021 da {main_path}...")

        # =====================================================
        # TRAIN MODE
        # =====================================================
        if mode == "train":
            train_dir = os.path.join(self.root, "train")
            
            if not os.path.exists(train_dir):
                raise FileNotFoundError(
                    f"Directory {train_dir} non trovata.\n"
                    f"Struttura attesa: {main_path}/train/*.png"
                )

            train_imgs = sorted(os.listdir(train_dir))
            
            if not train_imgs:
                raise FileNotFoundError(f"Nessuna immagine trovata in {train_dir}")

            t0 = time.time()

            self.slices += parallel_load(train_dir, train_imgs, img_size)
            self.labels += [0] * len(train_imgs)
            self.img_ids += [img.split(".")[0] for img in train_imgs]

            elapsed = time.time() - t0
            print(f"  ✓ Caricate {len(train_imgs)} immagini SANE in {elapsed:.2f}s")

        # =====================================================
        # TEST MODE
        # =====================================================
        else:
            normal_dir = os.path.join(self.root, "test", "normal")
            tumor_dir = os.path.join(self.root, "test", "tumor")
            annotation_dir = os.path.join(self.root, "test", "annotation")

            # Validazione path
            for p in [normal_dir, tumor_dir, annotation_dir]:
                if not os.path.exists(p):
                    raise FileNotFoundError(
                        f"Directory {p} non trovata.\n"
                        f"Struttura attesa:\n"
                        f"  {self.root}/test/normal/*.png\n"
                        f"  {self.root}/test/tumor/*.png\n"
                        f"  {self.root}/test/annotation/*_seg.png"
                    )

            normal_imgs = sorted(os.listdir(normal_dir))
            tumor_imgs = sorted(os.listdir(tumor_dir))
            
            if not normal_imgs or not tumor_imgs:
                raise FileNotFoundError(
                    f"Nessuna immagine trovata in {normal_dir} o {tumor_dir}"
                )

            # Mapping nome immagine → nome maschera
            # Supporta variazioni: img.png → img_seg.png oppure img_flair.png → img_seg.png
            tumor_masks = [file.replace("flair", "seg") for file in tumor_imgs]

            t0 = time.time()

            # Carica immagini
            self.slices += parallel_load(normal_dir, normal_imgs, img_size)
            self.slices += parallel_load(tumor_dir, tumor_imgs, img_size)

            # Maschere vuote per sani
            self.masks += [
                np.zeros((img_size, img_size), dtype=np.float32)
                for _ in range(len(normal_imgs))
            ]
            
            # Maschere reali (nearest neighbor per preservare bordi binari)
            self.masks += parallel_load(
                annotation_dir,
                tumor_masks,
                img_size,
                resample="nearest"
            )

            # Labels
            self.labels += [0] * len(normal_imgs) + [1] * len(tumor_imgs)
            
            all_imgs = normal_imgs + tumor_imgs
            self.img_ids += [file.split(".")[0] for file in all_imgs]

            elapsed = time.time() - t0
            print(f"  ✓ Caricate {len(normal_imgs)} immagini SANE in {elapsed:.2f}s")
            print(f"  ✓ Caricate {len(tumor_imgs)} immagini ANOMALE in {elapsed:.2f}s")

    def __getitem__(self, index: int) -> Dict:
        """
        Restituisce un campione.
        
        Args:
            index: indice campione
        
        Returns:
            Dict con chiavi:
            - 'img': torch.Tensor [1, H, W] normalizzato
            - 'label': int (0 o 1)
            - 'name': str (ID immagine)
            - 'mask': torch.Tensor [1, H, W] (solo test mode)
            - 'img_masked': torch.Tensor (solo se context_encoding=True)
        """
        img = self.slices[index]
        img = self.transform(img)
        
        label = self.labels[index]
        img_name = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {
                    "img": img,
                    "img_masked": img_masked,
                    "label": label,
                    "name": img_name
                }
            return {
                "img": img,
                "label": label,
                "name": img_name
            }
        
        else:  # TEST MODE
            mask_np = np.asarray(self.masks[index], dtype=np.float32)
            mask_np = (mask_np > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
            
            return {
                "img": img,
                "label": label,
                "name": img_name,
                "mask": mask_tensor
            }

    def __len__(self) -> int:
        """Numero totale di campioni."""
        return len(self.slices)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Ritorna statistiche del dataset.
        
        Returns:
            Dict con:
            - n_samples: numero campioni
            - n_healthy: numero campioni sani
            - n_anomalous: numero campioni anomali
            - anomaly_rate: percentuale anomali
            - img_shape: shape immagini
        """
        labels_arr = np.array(self.labels)
        n_healthy = (labels_arr == 0).sum()
        n_anomalous = (labels_arr == 1).sum()
        
        return {
            'n_samples': len(self),
            'n_healthy': int(n_healthy),
            'n_anomalous': int(n_anomalous),
            'anomaly_rate': float(n_anomalous / len(self)) if len(self) > 0 else 0,
            'img_shape': (self.res, self.res)
        }


# ==========================================================
# TRANSFORMAZIONI
# ==========================================================

def get_transforms(is_grayscale: bool = True) -> transforms.Compose:
    """
    Ritorna le transformazioni standard per BraTS2021.
    
    Normalizza nell'intervallo [-1, 1] senza data augmentation
    per preservare la coerenza anatomica delle immagini MRI.
    
    Args:
        is_grayscale: True per grayscale (1 canale), False per RGB
    
    Returns:
        torchvision.transforms.Compose con ToTensor + Normalize
    """
    mean_std = (0.5,) if is_grayscale else (0.5, 0.5, 0.5)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std, mean_std)
    ])
    
    return transform


# ==========================================================
# FACTORY
# ==========================================================

def get_dataset(
    dataset_name: str,
    data_root: str = "data",
    img_size: int = 64,
    mode: str = "train",
    context_encoding: bool = False
) -> BraTSDataset:
    """
    Factory per instanziare dataset.
    
    Args:
        dataset_name: "brats"
        data_root: percorso root (default "data")
        img_size: dimensione immagini (default 64)
        mode: "train" o "test" (default "train")
        context_encoding: abilita SSL (default False)
    
    Returns:
        BraTSDataset istanziato
    
    Raises:
        ValueError: se dataset_name non è "brats"
        FileNotFoundError: se le directory richieste non esistono
    """
    
    transform = get_transforms()
    
    if dataset_name.lower() == "brats":
        path = os.path.join(data_root, "BraTS2021")
        return BraTSDataset(
            path,
            img_size=img_size,
            transform=transform,
            mode=mode,
            context_encoding=context_encoding
        )
    
    raise ValueError(
        f"Dataset '{dataset_name}' non supportato. Usa 'brats'."
    )


# ==========================================================
# TEST DI INTEGRITÀ
# ==========================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TEST DATALOADER BRATS2021")
    print("="*70)
    
    try:
        # Test TRAIN
        print("\n[1/2] Caricamento TRAIN set...")
        train_ds = get_dataset("brats", data_root="data", mode="train")
        train_stats = train_ds.get_statistics()
        print(f"  ✓ Campioni: {train_stats['n_samples']}")
        print(f"  ✓ Sani: {train_stats['n_healthy']}")
        print(f"  ✓ Anomali: {train_stats['n_anomalous']}")
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        for batch in train_loader:
            print(f"  ✓ Batch shape: {batch['img'].shape}")
            break
        
        # Test TEST
        print("\n[2/2] Caricamento TEST set...")
        test_ds = get_dataset("brats", data_root="data", mode="test")
        test_stats = test_ds.get_statistics()
        print(f"  ✓ Campioni: {test_stats['n_samples']}")
        print(f"  ✓ Sani: {test_stats['n_healthy']}")
        print(f"  ✓ Anomali: {test_stats['n_anomalous']}")
        print(f"  ✓ Anomaly rate: {test_stats['anomaly_rate']:.2%}")
        
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        for batch in test_loader:
            print(f"  ✓ Batch img shape: {batch['img'].shape}")
            print(f"  ✓ Batch mask shape: {batch['mask'].shape}")
            break
        
        print("\n" + "="*70)
        print("✅ DATALOADER FUNZIONANTE!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("\nVerifica che i dati siano in:")
        print("  data/BraTS2021/train/")
        print("  data/BraTS2021/test/normal/")
        print("  data/BraTS2021/test/tumor/")
        print("  data/BraTS2021/test/annotation/")