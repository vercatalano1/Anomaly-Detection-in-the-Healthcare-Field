import os
import time
import numpy as np
from PIL import Image
from joblib import Parallel, delayed

import torch
from torch.utils import data
from torchvision import transforms


def parallel_load(img_dir, img_list, img_size, n_channel=1, resample="bilinear", verbose=0):
    """
    Carica le immagini in parallelo sfruttando la CPU multicore.
    Ridimensiona automaticamente le immagini a img_size x img_size.
    """
    mode = "L" if n_channel == 1 else "RGB"
    
    if resample == "bilinear":
        resample = Image.BILINEAR
    elif resample == "nearest":
        resample = Image.NEAREST
    else:
        raise Exception("Metodo di resample non valido")
        
    return Parallel(n_jobs=-1, verbose=verbose)(delayed(
        lambda file: Image.open(os.path.join(img_dir, file)).convert(mode).resize(
            (img_size, img_size), resample=resample))(file) for file in img_list)


class BraTSDataset(data.Dataset):
    """
    Dataloader definitivo per il dataset BraTS2021 (Risonanze Magnetiche Cerebrali).
    Gestisce maschere (ground truth) in formato Tensore e Data Augmentation.
    """
    def __init__(self, main_path, img_size=64, transform=None, mode="train", context_encoding=False):
        super(BraTSDataset, self).__init__()
        assert mode in ["train", "test"], "La modalità deve essere 'train' o 'test'"

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        
        # Mascheramento per Self-Supervised Learning (In-painting)
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print(f"\n[{mode.upper()}] Caricamento immagini BraTS2021 in corso...")
        
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print(f"Caricate {len(train_normal)} slice sane per il Train in {time.time() - t0:.2f}s")

        else:  # Modalità TEST
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            # Ricava il nome della maschera (Sostituisce flair con seg)
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            
            # Carica immagini
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            # Crea maschere vuote (zeri) per le immagini sane
            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            # Carica le maschere reali (Nearest Neighbor per non sfuocare i bordi binari)
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print(f"Caricate {len(test_l)} slice di Test ({len(test_normal)} sane, {len(test_abnormal)} tumorali) in {time.time() - t0:.2f}s")

    def __getitem__(self, index):
        img = self.slices[index]
        img = self.transform(img)

        label = self.labels[index]
        img_id = self.img_ids[index]

        if self.mode == "train":
            if self.random_mask is not None:
                img_masked = self.random_mask(img)
                return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
            else:
                return {'img': img, 'label': label, 'name': img_id}
        else:
            # Preparazione sicura della maschera in Tensore (Evita crash nel Dice Score)
            mask_np = np.array(self.masks[index])
            mask_np = (mask_np > 0).astype(np.float32) # Binarizza: 0 sfondo, 1 tumore
            mask_tensor = torch.tensor(mask_np).unsqueeze(0) # Forma finale: [1, 64, 64]
            
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask_tensor}

    def __len__(self):
        return len(self.slices)


# ==========================================
# FUNZIONI DI SUPPORTO E TRASFORMAZIONE
# ==========================================

def get_transforms(is_grayscale=True, mode="train"):
    """
    Applica la Data Augmentation in Train (ribaltamenti) e la normalizzazione rigorosa [-1, 1].
    """
    mean_std = (0.5,) if is_grayscale else (0.5, 0.5, 0.5)
    
    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5), # Evita overfitting
            transforms.RandomVerticalFlip(p=0.5),   # I cervelli (fette assiali) tollerano il flip
            transforms.ToTensor(),
            transforms.Normalize(mean_std, mean_std)
        ])
    else:
        # Nel TEST nessuna rotazione: i dati devono essere il ground truth assoluto
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_std, mean_std)
        ])
    return transform

def get_dataset(dataset_name, data_root="data", img_size=64, mode="train", context_encoding=False):
    """
    Factory pattern per l'inizializzazione del Dataset.
    """
    transform = get_transforms(is_grayscale=True, mode=mode)
    
    if dataset_name.lower() == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        return BraTSDataset(path, img_size, transform, mode, context_encoding)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato in questa configurazione.")


if __name__ == "__main__":
    # Test di integrità del codice
    print("\n=== TEST DATALOADER BRATS (TRAIN) ===")
    brats_train = get_dataset("brats", data_root="data", mode="train")
    train_loader = data.DataLoader(brats_train, batch_size=32, shuffle=True)
    
    for batch in train_loader:
        print(f"Batch Immagini (Train): {batch['img'].shape}") # Aspettato: [32, 1, 64, 64]
        break 
        
    print("\n=== TEST DATALOADER BRATS (TEST) ===")
    brats_test = get_dataset("brats", data_root="data", mode="test")
    test_loader = data.DataLoader(brats_test, batch_size=32, shuffle=False)
    
    for batch in test_loader:
        print(f"Batch Immagini (Test): {batch['img'].shape}")
        print(f"Batch Maschere (Test): {batch['mask'].shape}") # Aspettato: [32, 1, 64, 64]
        break