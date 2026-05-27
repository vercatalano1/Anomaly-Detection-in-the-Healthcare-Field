import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import get_dataset

# Configurazione stile grafico per la tesi
sns.set_theme(style="whitegrid", font_scale=1.1)

def plot_comprehensive_eda(dataset_name, dataset):
    """
    Analisi esplorativa avanzata per validazione dataset tesi.
    Focalizzata esclusivamente su BraTS2021.
    """
    print(f"\n{'='*20}\nANALISI: {dataset_name}\n{'='*20}")
    
    # Preleviamo i dati in modo efficiente
    imgs = torch.stack([dataset[i]['img'] for i in range(len(dataset))]).squeeze().numpy()
    labels = np.array(dataset.labels)
    
    # 1. Distribuzione Classi: Dimostrazione dello sbilanciamento (o equilibrio)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=labels, palette=['#2ca02c', '#d62728'])
    ax.set_xticklabels(['Sani', 'Anomali'])
    plt.title(f"Distribuzione Classi - {dataset_name}", fontweight='bold')
    plt.ylabel("Conteggio Immagini")
    plt.savefig(f"results/eda_{dataset_name}_dist.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Analisi Intensità: Verifica della normalizzazione
    mean_val = np.mean(imgs)
    std_val = np.std(imgs)
    print(f"Statistiche Pixel -> Media: {mean_val:.4f}, Std: {std_val:.4f}")
    
    plt.figure(figsize=(6, 4))
    sns.kdeplot(imgs.flatten(), fill=True, color='purple', alpha=0.3)
    plt.title(f"Densità Intensità Pixel: {dataset_name}", fontweight='bold')
    plt.xlabel("Range Normalizzato [-1, 1]")
    plt.savefig(f"results/eda_{dataset_name}_density.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 3. Analisi Anatomica: Media (Allineamento) e Varianza (Instabilità)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(np.mean(imgs, axis=0), cmap='bone')
    axes[0].set_title("Anatomia Media (Dataset Alignment)")
    axes[0].axis('off')
    
    axes[1].imshow(np.std(imgs, axis=0), cmap='jet')
    axes[1].set_title("Varianza Anatomica (Heatmap)")
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f"results/eda_{dataset_name}_anatomy.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Analisi specifica per BraTS (Segmentazione)
    # Estraiamo le maschere convertendo i Tensori [1, 64, 64] in array NumPy [64, 64]
    masks = np.stack([dataset[i]['mask'].numpy().squeeze() for i in range(len(dataset))])
    tumor_ratios = np.sum(masks, axis=(1, 2)) / (64*64)
    
    plt.figure(figsize=(6, 4))
    sns.histplot(tumor_ratios[tumor_ratios > 0], bins=30, kde=True, color='red')
    plt.title("Distribuzione Area Tumore (%)", fontweight='bold')
    plt.xlabel("Percentuale Copertura Tumore")
    plt.savefig(f"results/eda_{dataset_name}_tumor_ratio.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Area tumorale media: {np.mean(tumor_ratios[tumor_ratios > 0])*100:.2f}% della fetta.")

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    
    # Eseguiamo l'EDA SOLO per BraTS (Il nostro dataset definitivo)
    ds_brats = get_dataset("brats", data_root="data", mode="test")
    plot_comprehensive_eda("BraTS2021", ds_brats)