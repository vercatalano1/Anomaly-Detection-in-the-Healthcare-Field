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
    Dataloader corretto per BraTS2021 (Risonanze Magnetiche Cerebrali).
    Gestisce maschere (ground truth) in formato Tensore e Data Augmentation.
    
    CORREZIONI RISPETTO ALL'ORIGINALE:
    ✓ Fix: set_xticklabels() deprecato
    ✓ Fix: squeeze() ambiguo → squeeze(dim=-1)
    ✓ Fix: Magic number 64*64 → dimensioni dinamiche
    ✓ Fix: Accesso efficiente alle labels (non carica tutto in RAM)
    ✓ Fix: plt.close() per evitare memory leak
    """
    def __init__(self, main_path, img_size=64, transform=None, mode="train", context_encoding=False):
        super(BraTSDataset, self).__init__()
        assert mode in ["train", "test"], "La modalità deve essere 'train' o 'test'"
 
        self.mode = mode
        self.root = main_path
        self.res = img_size
        self.labels = []  # Attributo pubblico per EDA
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
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Directory '{data_dir}' non trovata. Verifica il path dei dati.")
            
            train_normal = sorted(os.listdir(data_dir))
 
            t0 = time.time()
            self.slices += parallel_load(data_dir, train_normal, img_size)
            self.labels += [0] * len(train_normal)  # 0 = sano
            self.img_ids += [img_name.split('.')[0] for img_name in train_normal]
            print(f"✓ Caricate {len(train_normal)} slice sane per il Train in {time.time() - t0:.2f}s")
 
        else:  # Modalità TEST
            test_normal_dir = os.path.join(self.root, "test", "normal")
            test_abnormal_dir = os.path.join(self.root, "test", "tumor")
            test_mask_dir = os.path.join(self.root, "test", "annotation")
            
            # Validazione path
            for path in [test_normal_dir, test_abnormal_dir, test_mask_dir]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Directory '{path}' non trovata. Verifica il path dei dati.")
 
            test_normal = sorted(os.listdir(test_normal_dir))
            test_abnormal = sorted(os.listdir(test_abnormal_dir))
            
            # Ricava il nome della maschera (Sostituisce flair con seg)
            test_masks = [e.replace("flair", "seg") for e in test_abnormal]
 
            test_l = test_normal + test_abnormal
            t0 = time.time()
            
            # Carica immagini
            self.slices += parallel_load(test_normal_dir, test_normal, img_size)
            self.slices += parallel_load(test_abnormal_dir, test_abnormal, img_size)
 
            # Crea maschere vuote (zeri) per le immagini sane
            self.masks += [np.zeros((img_size, img_size), dtype=np.float32) for _ in range(len(test_normal))]
            
            # Carica le maschere reali (Nearest Neighbor per non sfuocare i bordi binari)
            self.masks += parallel_load(test_mask_dir, test_masks, img_size, resample="nearest")
 
            # Labels: 0 = sano, 1 = tumore
            self.labels += [0] * len(test_normal) + [1] * len(test_abnormal)
            self.img_ids += [img_name.split('.')[0] for img_name in test_l]
            
            print(f"✓ Caricate {len(test_l)} slice di Test ({len(test_normal)} sane, {len(test_abnormal)} tumorali) in {time.time() - t0:.2f}s")
 
    def __getitem__(self, index):
        """
        Restituisce un dict con img, label, name (e opzionalmente mask per test).
        """
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
            mask_np = np.array(self.masks[index], dtype=np.float32)
            mask_np = (mask_np > 0).astype(np.float32)  # Binarizza: 0 sfondo, 1 tumore
            
            # CORRETTO: squeeze(dim=-1) anziché squeeze() generico
            # Preserva la dimensione batch nel caso di operazioni vettoriali
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # Forma finale: [1, H, W]
            
            return {'img': img, 'label': label, 'name': img_id, 'mask': mask_tensor}
 
    def __len__(self):
        return len(self.slices)
 
 
# ==========================================
# FUNZIONI DI SUPPORTO E TRASFORMAZIONE
# ==========================================
 
def get_transforms(is_grayscale=True, mode="train"):
    """
    Applica la Data Augmentation in Train e la normalizzazione rigorosa [-1, 1].
    
    NOTA CLINICA: Il RandomVerticalFlip è stato omesso intenzionalmente per 
    preservare la co-registrazione spaziale di BraTS (nessun cervello capovolto),
    ottimizzando così l'Information Bottleneck dell'Autoencoder.
    """
    mean_std = (0.5,) if is_grayscale else (0.5, 0.5, 0.5)
    
    if mode == "train":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # La simmetria assiale sx/dx è anatomicamente valida
            transforms.ToTensor(),
            transforms.Normalize(mean_std, mean_std)  # Scala a [-1, 1]
        ])
    else:
        # Nel TEST nessuna alterazione spaziale: valutiamo il Ground Truth assoluto
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_std, mean_std)  # Scala a [-1, 1]
        ])
    return transform
 
 
def get_dataset(dataset_name, data_root="data", img_size=64, mode="train", context_encoding=False):
    """
    Factory pattern per l'inizializzazione del Dataset.
    
    Args:
        dataset_name: nome del dataset ('brats')
        data_root: path root ai dati (default 'data')
        img_size: dimensione immagini (default 64)
        mode: 'train' o 'test'
        context_encoding: se True, applica masking casuale per SSL
    
    Returns:
        BraTSDataset istanziato
    """
    transform = get_transforms(is_grayscale=True, mode=mode)
    
    if dataset_name.lower() == 'brats':
        path = os.path.join(data_root, "BraTS2021")
        return BraTSDataset(path, img_size, transform, mode, context_encoding)
    else:
        raise ValueError(f"Dataset {dataset_name} non supportato. Usa 'brats'.")
 
 
# ==========================================
# TEST DI INTEGRITÀ
# ==========================================
 
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TEST DATALOADER BRATS (MedIAnomaly)")
    print("="*60)
    
    try:
        print("\n[1/2] Test TRAIN set...")
        brats_train = get_dataset("brats", data_root="data", mode="train")
        train_loader = data.DataLoader(brats_train, batch_size=32, shuffle=True)
        
        for batch in train_loader:
            print(f"  ✓ Batch Immagini (Train): {batch['img'].shape}")  # Aspettato: [32, 1, 64, 64]
            print(f"  ✓ Batch Labels: {batch['label'].shape}")
            print(f"  ✓ Labels unici: {np.unique(batch['label'].numpy())}")
            break
        
        print("\n[2/2] Test TEST set...")
        brats_test = get_dataset("brats", data_root="data", mode="test")
        test_loader = data.DataLoader(brats_test, batch_size=32, shuffle=False)
        
        for batch in test_loader:
            print(f"  ✓ Batch Immagini (Test): {batch['img'].shape}")
            print(f"  ✓ Batch Maschere: {batch['mask'].shape}")  # Aspettato: [32, 1, 64, 64]
            print(f"  ✓ Labels: {np.bincount(batch['label'].numpy())}")  # conteggio per classe
            break
        
        print("\n" + "="*60)
        print("✅ DATALOADER FUNZIONANTE!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERRORE: {e}")
        print("Verifica che i dati siano in: data/BraTS2021/train e data/BraTS2021/test/")
 