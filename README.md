# Anomaly Detection in the Healthcare Field
### Rilevamento e localizzazione non supervisionata di tumori cerebrali su risonanze magnetiche (BraTS2021)

> Studio comparativo di tre paradigmi di *anomaly detection* — Machine Learning classico,
> Deep Learning ricostruttivo e Transfer Learning — per l'identificazione
> e la localizzazione di tumori cerebrali a partire da slice MRI, addestrati **esclusivamente su
> immagini sane**. 

---

## 1. Obiettivo e motivazioni

Nella diagnostica per immagini, la classificazione supervisionata richiede grandi quantità di dati etichettati per ogni possibile variante patologica: un vincolo spesso non realistico per patologie rare, morfologicamente eterogenee o semplicemente perché non è pensabile raccogliere ground truth per ogni possibile anomalia. L'*anomaly detection* non supervisionata affronta il problema in modo diverso: il modello apprende esclusivamente la distribuzione del tessuto **sano**, e qualunque scostamento significativo da tale distribuzione viene trattato come potenziale anomalia.

Questo progetto confronta, a parità di dataset e protocollo sperimentale, **tre paradigmi** di anomaly detection non supervisionata applicati a slice MRI cerebrali:

1. **Isolation Forest** — baseline classica di machine learning su pixel grezzi
2. **CNN Autoencoder** — approccio generativo/ricostruttivo (loss ibrida MSE+L1, denoising gaussiano coarse, post-processing morfologico)
3. **PatchCore** — transfer learning da rete pre-addestrata su ImageNet, memory bank gerarchica + KNN (Roth et al., 2022)

L'obiettivo non è solo individuare l'immagine anomala (**image-level detection**), ma anche localizzare la lesione a livello di singolo pixel (**pixel-level segmentation**), usando le maschere di segmentazione **esclusivamente in fase di valutazione finale**, mai durante l'addestramento o la selezione del modello.

---

## 2. Dataset

Il dataset utilizzato è **BraTS2021** (Brain Tumor Segmentation Challenge 2021), riorganizzato secondo il formato del framework **MedIAnomaly**, che garantisce riproducibilità scientifica e un confronto equo tra i diversi paradigmi.

### 2.1 Struttura su disco
```
BraTS2021/
├── train/                  # esclusivamente slice sane (label = 0)
└── test/
    ├── normal/              # slice sane (label = 0)
    ├── tumor/               # slice con tumore (label = 1)
    └── annotation/          # maschere di segmentazione (solo per le slice tumor)
```

> **Nota:** la cartella `BraTS2021/` non è inclusa nel repository (vedi `.gitignore`) per motivi
> di dimensione e licenza dei dati. Va scaricata separatamente e posizionata nella root del progetto.

### 2.2 Composizione dei dati

| Split | Immagini | Pazienti |
|---|---:|---:|
| Train (sano) | 4.211 | 933 |
| Test — normal | 828 | 175 |
| Test — tumor | 1.948 | 199 |
| Annotation (maschere) | 1.948 | 199 |

- **Risoluzione originale:** 208×208 px (unica dimensione presente, 8.935 immagini totali)
- **Risoluzione di lavoro:** ridimensionata a **64×64 px** in fase di caricamento (dataloader)
- **Canali:** 1 (scala di grigi)
- **Range dei valori:** normalizzato in **[0, 1]**
- **Controllo data leakage:** nessuna sovrapposizione di pazienti tra `train` e i due split di `test` (`TRAIN ∩ TEST NORMAL = 0`, `TRAIN ∩ TEST TUMOR = 0`). Una sovrapposizione di 174 pazienti tra `TEST NORMAL` e `TEST TUMOR` è invece attesa: sono pazienti oncologici con slice sane in alcune sezioni assiali e slice tumorali in altre.

### 2.3 Slice per paziente

| Split | Min | Max | Media | Mediana | Std |
|---|---:|---:|---:|---:|---:|
| Train | 1 | 14 | 4.51 | 4.00 | 2.69 |
| Test normal | 1 | 14 | 4.73 | 4.00 | 2.87 |
| Test tumor | 1 | 14 | 9.79 | 10.00 | 3.12 |

### 2.4 Statistiche di intensità e sovrapposizione delle distribuzioni

L'analisi esplorativa (`src/data_analysis/eda.py`) evidenzia una significativa sovrapposizione tra intensità sane e tumorali, motivando la difficoltà del problema:

| Metrica | Normal | Tumor |
|---|---:|---:|
| Mean intensity | 30.25 | 36.08 |
| Std intensity | 45.70 | 51.61 |
| p95 intensity | 113.91 | 128.18 |
| Nonzero fraction | 0.330 | 0.356 |

- **Distanza di Wasserstein (globale):** 5.77
- **Distanza di Wasserstein (pixel non-zero):** 12.39
- **Area media della lesione:** 4.46% dei pixel per slice (mediana 4.03%, range 0.35%–14.89%)
- **PCA esplorativa (2 componenti, campione bilanciato N=1500):** varianza spiegata cumulativa ≈ 44.3% (PC1 = 30.7%, PC2 = 13.7%) — le due classi non sono linearmente separabili nello spazio dei pixel grezzi, giustificando l'adozione di rappresentazioni apprese (feature learning).

---

## 3. Protocollo sperimentale

Per garantire un confronto rigoroso e privo di *data leakage*, tutti i modelli seguono lo stesso protocollo:

- **Training:** esclusivamente immagini sane (`train/`)
- **Validation** (15% delle immagini sane di train, split fisso con seed 42): usata per *early stopping*, model selection e calibrazione delle soglie — **mai** il test set
- **Test:** sano + tumorale, usato **una sola volta** per la valutazione finale
- **Soglia image-level:** 95° percentile degli score di anomalia calcolato sul validation set sano
- **Soglia pixel-level:** 99° percentile delle anomaly map calcolato sul validation set sano
- Nessun tuning di soglia sul test set, nessuna informazione tumorale (label o maschera) utilizzata in training o validation
- Seed fisso (`SEED = 42`) per la riproducibilità di split, inizializzazione dei pesi e sampling

---

## 4. Modelli implementati

### 4.1 Isolation Forest — baseline classica

Modello di ensemble basato sull'isolamento casuale dei punti in uno spazio ad alta dimensionalità. Le immagini vengono appiattite (`64×64 → 4096`) e standardizzate (`StandardScaler` fittato solo sul train). `contamination="auto"`, soglia decisionale nativa a 0, nessun tuning.
Rappresenta il limite del ML classico: efficienza computazionale estrema ma nessuna capacità di localizzazione spaziale (perdita totale della struttura 2D con il flattening).

### 4.2 CNN Autoencoder — approccio generativo/ricostruttivo

Autoencoder convoluzionale, addestrato a ricostruire immagini sane, con loss ibrida MSE+L1, denoising gaussiano coarse in input e post-processing morfologico sulle anomaly map. Lo score di anomalia a livello immagine è l'errore quadratico medio di ricostruzione (MSE); la mappa di anomalia pixel-level è l'errore assoluto `|input - reconstruction|`.

### 4.3 PatchCore — transfer learning + memory bank

Basato su [Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265). Nessun training è eseguito: si sfrutta una ResNet-18 **pre-addestrata su ImageNet** (l'input single-channel viene replicato su 3 canali) per estrarre feature locali gerarchiche da `layer1` e `layer2`. Le feature estratte dalle immagini sane di train formano una **Memory Bank**, compressa tramite coreset subsampling casuale (5%). In fase di test, lo score di anomalia per ogni patch è la distanza euclidea (KNN, K=1) dalla patch più vicina in memoria; la mappa risultante viene smussata con un filtro gaussiano (`sigma=2.0`).

---

## 5. Struttura del repository

```
├── src/
│   ├── data_analysis/
│   │   ├── dataloader.py            # Dataset PyTorch (BraTSDataset) + factory get_dataset()
│   │   ├── eda.py                   # Analisi esplorativa completa + generazione figure
│   │   ├── inspect_brats.py         # Ispezione strutturale del dataset grezzo (file, dimensioni, estensioni)
│   │   └── validate_dataloader.py   # Validazione automatica di shape, range, NaN/Inf, batch PyTorch
│   │
│   ├── ml/
│   │   └── ml_baseline.py           # Isolation Forest (baseline ML pura, image-level)
│   │
│   ├── dl/
│   │   ├── cnn_ae/
│   │   │   ├── 3.post_processing.py # CNN Autoencoder (MSE+L1, denoising, post-processing)
│   │   │
│   │   └── PatchCore/
│   │       └── 1.baseline.py        # PatchCore (ResNet-18 + memory bank)
│   │
│   ├── collect_results.py           # Aggregazione automatica delle metriche 
│   ├── final_plot.py                # Grafici comparativi finali 
│   ├── compare_heatmaps.py          # Griglia orizzontale di confronto qualitativo delle heatmaps
│   ├── patient_level.py             # Aggregazione slice-level → patient-level (regola del max score)
│   └── confidence.py                # Intervalli di confidenza bootstrap (95%, N=2000) + DeLong test pairwise
│
├── results/                          # Output di ciascun esperimento (metriche, report, figure)
│   ├── eda/                          # Report e figure dell'analisi esplorativa
│   ├── ml_baseline/                  # Isolation Forest
│   ├── cnn_autoencoder_/             # Varianti del CNN Autoencoder (una cartella per variante)
│   ├── patchcore/                    # PatchCore
│   └── summary/                      # Tabelle e grafici comparativi finali, bootstrap CI, patient-level
│                      
├── README.md
└── .gitignore
```

---

## 6. Risultati

Tutti i valori sono estratti automaticamente da `src/collect_results.py` a partire dai CSV di ciascun esperimento (`results/summary model_comparison.csv`). 

### 6.1 Confronto principale — image-level e pixel-level

| Modello | Img AUROC | Img AP | Img F1 | Img Sens. | Img Spec. | Pixel AUROC | Pixel Dice | Tempo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | 0.670 | 0.794 | 0.489 | 0.348 | 0.827 | — | — | **0.6 s** |
| CNN Autoencoder | 0.799 | 0.902 | 0.547 | 0.386 | 0.940 | 0.909 | **0.375** | 13.2 min |
| **PatchCore** | **0.904** | **0.959** | **0.775** | **0.646** | **0.952** | **0.956** | **0.441** | 3.1 min |

### 6.2 Validazione statistica (Bootstrap IC 95% & Test di DeLong)
- **Bootstrap (2.000 resample):** PatchCore dimostra intervalli di confidenza stretti e nettamente separati dalle altre architetture (AUROC IC 95% = `[0.892, 0.914]`).
- **Test di DeLong pairwise:** PatchCore supera significativamente sia l'Isolation Forest ($\Delta\text{AUROC} = +0.233$, $p < 0.001$) sia il CNN Autoencoder ($\Delta\text{AUROC} = +0.104$, $p < 0.001$), confermando la superiorità statistica strutturale.

### 6.3 Valutazione patient-level

Oltre alla valutazione slice-level, `src/patient_level.py` aggrega gli score alla granularità del paziente secondo la regola clinica `patient_score = max(anomaly_score)` sulle slice del paziente (`patient_label = 1` se almeno una slice è tumorale), ricalcolando AUROC e AP a livello paziente per tutti i modelli compatibili. Risultati numerici e grafici sono in `results/summary/patient_level/`.

### 6.4 Grafici comparativi finali

Generati da `src/final_plot.py` in `results/summary/final_comparison/`:
- `image_level_performance.png` / `pixel_level_performance.png`
- `slice_level_roc_curves.png` & `slice_level_pr_curves.png`
- `pixel_dice_boxplot.png` (distribuzione del Dice per-slice)
- `detection_localization_tradeoff.png`
- `training_computational_cost.png`

---

## 7. Analisi critica

- **PatchCore** eccelle sia in detection sia in localizzazione grazie a feature semantiche pre-addestrate su ImageNet. Il vantaggio è statisticamente blindato dai test di DeLong e dai bootstrap CI.
- Il **CNN Autoencoder** offre un'ottima ricostruzione e localizzazione pixel-level (Pixel AUROC 0.909, Dice 0.375), ma sconta i limiti dell'addestramento da zero su un dataset ridotto a 64×64 pixel.
- **Isolation Forest** offre un'efficienza computazionale imbattibile (< 1 s), ma non è in grado di produrre localizzazione spaziale.

---

## 8. Requisiti e installazione

```bash
python >= 3.10
```

Pacchetti principali:

```
torch
torchvision
scikit-learn
numpy
pandas
matplotlib
seaborn
scipy
pillow
joblib
```

Installazione rapida (consigliato l'uso di un ambiente virtuale):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn scipy pillow joblib
```

> Per l'addestramento dei modelli deep (CNN Autoencoder e PatchCore) è fortemente consigliata una GPU CUDA; tutti gli script rilevano automaticamente il device disponibile
> (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## 9. Come riprodurre gli esperimenti

```bash
# 1. Analisi esplorativa del dataset
python src/data_analysis/eda.py

# 2. Dataloader e sua validazione
python src/data_analysis/dataloader.py
python src/data_analysis/validate_dataloader.py

# 3. Esecuzione modelli
python src/ml/ml_baseline.py
python src/dl/cnn_ae/3.post_processing.py
python src/dl/PatchCore/1.baseline.py

# 4. Aggregazione risultati e grafici comparativi finali
python src/collect_results.py
python src/final_plot.py

# 5. Estensioni statistiche e cliniche
python src/confidence.py        
python src/patient_level.py   
python src/compare_heatmaps.py   
```

Ogni script scrive i propri output (report `.txt`, CSV con le metriche, figure `.png`) in una sottocartella dedicata di `results/`, così da poter rieseguire un singolo esperimento senza impattare gli altri.

---

## 10. Limitazioni note

- **Elaborazione 2D slice-per-slice:** il volume MRI 3D viene trattato come una sequenza indipendente di slice assiali, con conseguente perdita della continuità volumetrica lungo l'asse Z.
- **Downsampling a 64×64:** riduce il costo computazionale ma comprime dettagli fini della lesione, con possibile impatto sulla localizzazione pixel-level (soprattutto per lesioni piccole, minoranza nella distribuzione delle aree tumorali osservata in EDA).
- **Sensibilità della soglia P99 (pixel-level):** calibrata sulla distribuzione dell'errore sano in validation; non necessariamente ottimale rispetto a metriche come il Dice score.


---

## 11. Sviluppi futuri

- Estensione a modelli **3D** (3D-CNN, 3D Autoencoder) per sfruttare la continuità volumetrica tra slice adiacenti.
- Adozione di **modelli generativi di frontiera** (Diffusion Models, Masked Autoencoders) come paradigma ricostruttivo alternativo al CNN Autoencoder classico.
- **Threshold-sweep** sistematico sulla soglia pixel-level per ottimizzare esplicitamente il Dice score, invece di fissare a priori il percentile P99.
- Validazione su **coorti esterne** come altri dataset BraTS o altre patologie oncologiche cerebrali, per verificare la generalizzazione dei risultati.

---
