import os
import time
import json
import numpy as np
from PIL import Image
from joblib import Parallel, delayed

import torch
from torch.utils import data
from torchvision import transforms


def parallel_load(img_dir, img_list, img_size, n_channel=1, resample="bilinear", verbose=0):
    """
    Carica le immagini in parallelo sfruttando la CPU multicore.
    Ridimensiona automaticamente le immagini a img_size x img_size (es. 64x64).
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


class RSNADataset(data.Dataset):
    """
    Dataloader per il dataset RSNA (Radiografie Toraciche).
    Utilizza il file data.json pre-impostato dal benchmark MedIAnomaly.
    """
    def __init__(self, main_path, img_size=64, transform=None, mode="train", context_encoding=False):
        super(RSNADataset, self).__init__()
        assert mode in ["train", "test"], "La modalità deve essere 'train' o 'test'"
        
        self.root = main_path
        self.labels = []
        self.img_id = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        
        # Mascheramento per Self-Supervised Learning (es. CutPaste/AnatPaste)
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        # Carica il dizionario delle suddivisioni ufficiali del benchmark
        with open(os.path.join(main_path, "data.json")) as f:
            data_dict = json.load(f)

        print(f"[{mode.upper()}] Caricamento immagini RSNA in corso...")
        if mode == "train":
            train_normal = data_dict["train"]["0"] # "0" indica immagini sane
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), train_normal, img_size)
            self.labels += len(train_normal) * [0]
            self.img_id += [img_name.split('.')[0] for img_name in train_normal]
            print("Caricate {} immagini sane in {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # modalità test
            test_normal = data_dict["test"]["0"]
            test_abnormal = data_dict["test"]["1"] # "1" indica presenza di anomalia
            test_l = test_normal + test_abnormal
            
            t0 = time.time()
            self.slices += parallel_load(os.path.join(self.root, "images"), test_l, img_size)
            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_id += [img_name.split('.')[0] for img_name in test_l]
            print("Caricate {} immagini di test ({} sane, {} anomale) in {:.3f}s".format(
                len(test_l), len(test_normal), len(test_abnormal), time.time() - t0))

    def __getitem__(self, index):
        img = self.slices[index]
        label = self.labels[index]
        img = self.transform(img)
        img_id = self.img_id[index]

        if self.random_mask is not None:
            img_masked = self.random_mask(img)
            return {'img': img, 'label': label, 'name': img_id, 'img_masked': img_masked}
        else:
            return {'img': img, 'label': label, 'name': img_id}

    def __len__(self):
        return len(self.slices)


class BraTSDataset(data.Dataset):
    """
    Dataloader per il dataset BraTS2021 (Risonanze Magnetiche Cerebrali 2D Slices).
    Gestisce il caricamento delle maschere (ground truth) per il task di Segmentazione (AnoSeg).
    """
    def __init__(self, main_path, img_size=64, transform=None, mode="train", context_encoding=False):
        super(BraTSDataset, self).__init__()
        assert mode in ["train", "test"]

        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []
        self.masks = []
        self.img_ids = []
        self.slices = []
        self.transform = transform if transform is not None else lambda x: x
        
        if context_encoding:
            self.random_mask = transforms.RandomErasing(p=1., scale=(0.024, 0.024), ratio=(1., 1.), value=-1)
        else:
            self.random_mask = None

        print(f"[{mode.upper()}] Caricamento immagini BraTS2021 in corso...")
        if mode == "train":
            data_dir = os.path.join(self.root, "train")
            train_normal = os.listdir(data_dir)

            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print("Caricate {} slice sane in {:.3f}s".format(len(train_normal), time.time() - t0))

        else:  # modalità test
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")

            test_normal = os.listdir(test_normal_dir)
            test_abnormal = os.listdir(test_abnormal_dir)
            # Ricava il nome della maschera sostituendo 'flair' con 'seg'
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]

            test_l = test_normal + test_abnormal
            t0 = time.time()
            
            # Carica immagini e maschere
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)

            # Crea maschere vuote (tutti zeri) per le immagini sane
            self.masks += len(test_normal) * [np.zeros((img_size, img_size))]
            # Carica le maschere reali per i tumori (usando resample nearest per non sfuocare i bordi binari)
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")

            self.labels += len(test_normal) * [0] + len(test_abnormal) * [1]
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            print("Caricate {} slice di test ({} sane, {} tumorali) in {:.3f}s".format(
                len(test_l), len(test_normal), len(test_abnormal), time.time() - t0))

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
            # Per il test su BraTS, ritorniamo anche la 'mask' per calcolare il Dice Score!
            mask = np.array(self.masks[index])
            mask = (mask > 0).astype(np.uint8) # Binarizza la maschera (0 sfondo, 1 tumore)
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask}

    def __len__(self):
        return len(self.slices)


# ==========================================
# FUNZIONI DI SUPPORTO E TRASFORMAZIONE
# ==========================================

def get_transform(is_grayscale=True):
    """
    Applica le trasformazioni standard: converte in Tensore e normalizza.
    La normalizzazione a 0.5 (mean e std) scala i pixel dal range [0, 1] a [-1, 1],
    che è la prassi standard per far convergere meglio gli Autoencoder.
    """
    mean_std = (0.5,) if is_grayscale else (0.5, 0.5, 0.5)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_std, mean_std)
    ])
    return transform

def get_dataset(dataset_name, data_root="data", img_size=64, mode="train", context_encoding=False):
    """
    Factory pattern per caricare in modo pulito il dataset richiesto.
    """
    transform = get_transform(is_grayscale=True) # Medical imaging = 1 channel (Grigio)
    
    if dataset_name.lower() == 'rsna':
        path = os.path.join(data_root, "RSNA")
        return RSNADataset(path, img_size, transform, mode, context_encoding)
    elif dataset_name.lower() == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        return BraTSDataset(path, img_size, transform, mode, context_encoding)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato in questa tesi.")



if __name__ == "__main__":
    # Testiamo RSNA
    print("=== TEST DATALOADER RSNA ===")
    rsna_dataset = get_dataset("rsna", data_root="data", mode="train")
    # Carichiamo 32 immagini alla volta
    rsna_loader = data.DataLoader(rsna_dataset, batch_size=32, shuffle=True)
    
    for batch in rsna_loader:
        img_tensor = batch['img']
        print(f"Forma del batch RSNA: {img_tensor.shape}") # Deve essere [32, 1, 64, 64]
        break # Ne stampiamo solo uno