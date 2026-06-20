import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter

from dataloader import get_dataset

sns.set_theme(style="whitegrid", font_scale=1.1)


# ==========================================================
# UTILS
# ==========================================================
def balanced_indices(labels, n):
    labels = np.array(labels)

    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    idx0 = np.random.choice(idx0, min(n, len(idx0)), replace=False)
    idx1 = np.random.choice(idx1, min(n, len(idx1)), replace=False)

    return idx0, idx1


def extract_pixels(dataset, indices):
    vals = []
    for i in indices:
        img = dataset[i]["img"].detach().cpu().numpy()
        vals.append(img.flatten())
    return np.concatenate(vals)


def extract_images(dataset, indices):
    imgs = []
    for i in indices:
        img = dataset[i]["img"].detach().cpu().numpy()[0]
        imgs.append(img)
    return np.array(imgs)


# ==========================================================
# 1. CLASS DISTRIBUTION (FONDAMENTALE)
# ==========================================================
def plot_class_distribution(train, test, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i, ds in enumerate([train, test]):
        labels = np.array(ds.labels)
        unique, counts = np.unique(labels, return_counts=True)

        ax[i].bar(unique, counts,
                  color=["#2ca02c", "#d62728"],
                  edgecolor="black")

        ax[i].set_xticks([0, 1])
        ax[i].set_xticklabels(["Sani", "Anomali"])
        ax[i].set_title("Train" if i == 0 else "Test", fontweight="bold")

        for u, c in zip(unique, counts):
            ax[i].text(u, c + 5, str(c), ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/01_class_distribution.png", dpi=300)
    plt.close()


# ==========================================================
# 2. INTENSITY DISTRIBUTION + WASSERSTEIN (SEPARABILITÀ)
# ==========================================================
def plot_intensity(test, out_dir, n=200):

    idx0, idx1 = balanced_indices(test.labels, n)

    healthy = extract_pixels(test, idx0)
    tumor = extract_pixels(test, idx1)

    wd = wasserstein_distance(healthy, tumor)

    plt.figure(figsize=(7, 5))

    sns.kdeplot(healthy, fill=True, color="#2ca02c", alpha=0.4, label="Sani")
    sns.kdeplot(tumor, fill=True, color="#d62728", alpha=0.4, label="Anomali")

    plt.title(f"Distribuzione Intensità (Wasserstein = {wd:.4f})",
              fontweight="bold")

    plt.xlabel("Intensità normalizzata")
    plt.ylabel("Densità")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/02_intensity.png", dpi=300)
    plt.close()

    print(f"[INFO] Wasserstein distance: {wd:.4f}")


# ==========================================================
# 3. ANALISI ANATOMICA (STRUTTURALE)
# ==========================================================
def plot_anatomical(test, out_dir, n=150):

    idx0, idx1 = balanced_indices(test.labels, n)

    healthy = extract_images(test, idx0)
    tumor = extract_images(test, idx1)

    m0 = healthy.mean(axis=0)
    m1 = tumor.mean(axis=0)
    diff = m1 - m0

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    ax[0].imshow(m0, cmap="bone")
    ax[0].set_title("Sani")
    ax[0].axis("off")

    ax[1].imshow(m1, cmap="bone")
    ax[1].set_title("Anomali")
    ax[1].axis("off")

    vmax = np.abs(diff).max()
    ax[2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[2].set_title("Differenza")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/03_anatomical.png", dpi=300)
    plt.close()


# ==========================================================
# 4. HEATMAP FREQUENZA TUMORE (LOCALIZZAZIONE)
# ==========================================================
def plot_frequency(test, out_dir):

    idx = np.where(np.array(test.labels) == 1)[0]

    masks = []
    for i in idx:
        m = test[i]["mask"].detach().cpu().numpy()[0]
        masks.append(m)

    freq = np.array(masks).mean(axis=0)

    # smoothing per visualizzazione più chiara
    freq = gaussian_filter(freq, sigma=1)

    plt.figure(figsize=(6, 6))
    plt.imshow(freq, cmap="hot")
    plt.colorbar(label="Frequenza tumore")

    plt.title("Heatmap Frequenza Tumorale")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/04_frequency.png", dpi=300)
    plt.close()


# ==========================================================
# 5. TUMOR AREA (HIST + BOXPLOT)
# ==========================================================
def plot_tumor_area(dataset, output_dir="results"):
    """
    5. Statistiche maschere tumorali — distribuzione area e boxplot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("  → Calcolo statistiche maschere...")
    
    labels_arr = np.array(dataset.labels)
    anomalous_indices = np.where(labels_arr == 1)[0]
    
    tumor_areas = []
    for idx in anomalous_indices:
        sample = dataset[int(idx)]
        mask = sample['mask'].numpy()[0]
        tumor_area = (mask > 0.5).sum() / mask.size * 100
        tumor_areas.append(tumor_area)
    
    tumor_areas = np.array(tumor_areas)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Istogramma
    ax = axes[0]
    counts, bins, patches = ax.hist(tumor_areas, bins=30, color='#d62728', alpha=0.7, edgecolor='black', linewidth=1)
    ax.axvline(tumor_areas.mean(), color='darkred', linestyle='--', linewidth=2.5, label=f'Media: {tumor_areas.mean():.2f}%')
    ax.axvline(np.median(tumor_areas), color='orange', linestyle=':', linewidth=2.5, label=f'Mediana: {np.median(tumor_areas):.2f}%')
    ax.set_xlabel("Percentuale Area Tumorale (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Frequenza", fontsize=11, fontweight='bold')
    ax.set_title("Distribuzione Area Tumorale", fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    
    # Boxplot
    ax = axes[1]
    bp = ax.boxplot(tumor_areas, vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor('#d62728')
    bp['boxes'][0].set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(linewidth=1.5)
    for cap in bp['caps']:
        cap.set(linewidth=1.5)
    for median in bp['medians']:
        median.set(color='darkred', linewidth=2)
    
    ax.set_ylabel("Percentuale Area Tumorale (%)", fontsize=11, fontweight='bold')
    ax.set_title("Boxplot Area Tumorale", fontweight='bold', fontsize=12)
    ax.set_xticklabels(['Anomali'])
    ax.grid(alpha=0.3, axis='y')
    
    # Statistiche testuali sul grafico
    stats_text = (f"n = {len(tumor_areas)}\n"
                 f"Min: {tumor_areas.min():.2f}%\n"
                 f"Max: {tumor_areas.max():.2f}%\n"
                 f"Media: {tumor_areas.mean():.2f}%\n"
                 f"Mediana: {np.median(tumor_areas):.2f}%\n"
                 f"Std: {tumor_areas.std():.2f}%")
    ax.text(1.35, tumor_areas.mean(), stats_text, fontsize=9, 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_tumor_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 05_tumor_statistics.png")


# ==========================================================
# 6. OVERLAY SEGMENTATION (EVIDENZA VISIVA)
# ==========================================================
def plot_overlay(test, out_dir, n=6):

    labels = np.array(test.labels)

    idx0 = np.random.choice(np.where(labels == 0)[0], n//2, replace=False)
    idx1 = np.random.choice(np.where(labels == 1)[0], n//2, replace=False)

    idx = np.concatenate([idx0, idx1])

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax = ax.flatten()

    for i, j in enumerate(idx):

        img = test[j]["img"].detach().cpu().numpy()[0]
        mask = test[j]["mask"].detach().cpu().numpy()[0]
        name = test[j]["name"]

        ax[i].imshow(img, cmap="gray")

        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0.5] = [1, 0, 0, 0.4]

        ax[i].imshow(overlay)

        ax[i].set_title(name)
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/06_overlay.png", dpi=300)
    plt.close()


# ==========================================================
# 7. PCA FEATURE SPACE (SEPARABILITY ANALYSIS)
# ==========================================================
def plot_pca(test, out_dir):

    X = np.array([test[i]["img"].numpy().flatten() for i in range(len(test))])
    y = np.array(test.labels)

    X_pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(6, 5))

    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
                s=6, label="Sani", alpha=0.6)

    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
                s=6, label="Anomali", alpha=0.6)

    plt.title("Proiezione PCA")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{out_dir}/07_pca.png", dpi=300)
    plt.close()


# ==========================================================
# 7. REPORT FINALE
# ==========================================================
def generate_eda_report(train, test, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    tr = np.array(train.labels)
    te = np.array(test.labels)

    test_anom = (te == 1).sum()
    test_sani = (te == 0).sum()

    path = os.path.join(out_dir, "REPORT_EDA.txt")

    with open(path, "w", encoding="utf-8") as f:

        f.write("="*80 + "\n")
        f.write("ANALISI EDA - BraTS2021\n")
        f.write("Task: Anomaly Detection in Imaging Medico\n")
        f.write("="*80 + "\n\n")

        f.write("COMPOSIZIONE DATASET\n")
        f.write(f"Train: {len(train)} (solo sani)\n")
        f.write(f"Test: {len(test)} ({test_sani} sani, {test_anom} anomali)\n\n")

        f.write("RISULTATI CHIAVE\n")
        f.write("- Separabilità tra classi nelle distribuzioni di intensità\n")
        f.write("- Presenza di strutture anatomiche differenti nei casi patologici\n")
        f.write("- Elevata variabilità della dimensione delle lesioni\n\n")

        f.write("OSSERVAZIONI PRINCIPALI\n")
        f.write("- Le distribuzioni delle intensità mostrano una parziale separazione tra immagini sane e patologiche.\n")
        f.write("- Le immagini medie evidenziano differenze strutturali associate alla presenza della massa tumorale.\n")
        f.write("- La mappa di frequenza conferma che le anomalie possono comparire in regioni differenti del cervello.\n")
        f.write("- La variabilità dell'area tumorale suggerisce la necessità di modelli robusti a lesioni di dimensioni differenti.\n")
        f.write("- La proiezione nello spazio delle componenti principali (PCA) mostra una parziale separazione tra immagini sane e patologiche.\n\n")

        f.write("MOTIVAZIONE DATASET\n")
        f.write("- BraTS2021 rappresenta uno dei benchmark più utilizzati nell'imaging cerebrale.\n")
        f.write("- Le lesioni tumorali sono annotate mediante maschere di segmentazione affidabili.\n")
        f.write("- La disponibilità di immagini sane e patologiche rende il dataset particolarmente adatto allo studio dell'anomaly detection.\n")
        f.write("- L'elevata variabilità anatomica e patologica permette di valutare la robustezza dei modelli.\n")
        f.write("- La struttura del train set, costituito esclusivamente da esempi normali, è coerente con il paradigma one-class learning utilizzato da Isolation Forest e Autoencoder.\n\n")

        f.write("="*80 + "\n")

# ==========================================================
# MAIN
# ==========================================================
def run_eda():

    out_dir = "results/eda"
    os.makedirs(out_dir, exist_ok=True)

    print("\n==============================")
    print("EDA FINALE - BRA TS2021")
    print("==============================")

    train = get_dataset("brats", mode="train")
    test = get_dataset("brats", mode="test")

    print("[1/7] Class distribution")
    plot_class_distribution(train, test, out_dir)

    print("[2/7] Intensity separability")
    plot_intensity(test, out_dir)

    print("[3/7] Anatomical comparison")
    plot_anatomical(test, out_dir)

    print("[4/7] Tumor localization")
    plot_frequency(test, out_dir)

    print("[5/7] Tumor area variability")
    plot_tumor_area(test, out_dir)

    print("[6/7] Visual segmentation examples")
    plot_overlay(test, out_dir)

    print("[7/7] PCA space")
    plot_pca(test, out_dir)

    print("\n[REPORT] Generazione report...")
    generate_eda_report(train, test, out_dir)
    
    print("\n✅ EDA COMPLETATO")
    print(f"Output: {out_dir}/")


if __name__ == "__main__":
    run_eda()
