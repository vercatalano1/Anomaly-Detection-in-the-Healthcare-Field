# Anomaly Detection in the Healthcare Field
### Rilevamento e localizzazione non supervisionata di tumori cerebrali su risonanze magnetiche (BraTS2021)

> Studio comparativo di tre paradigmi di *anomaly detection* — Machine Learning classico,
> Deep Learning ricostruttivo e Transfer Learning — per l'identificazione
> e la localizzazione di tumori cerebrali a partire da slice MRI, addestrati **esclusivamente su
> immagini sane**. 

---

## 1. Obiettivo e motivazioni

Nella diagnostica per immagini, la classificazione supervisionata richiede grandi quantità di dati
etichettati per ogni possibile variante patologica: un vincolo spesso non realistico per patologie
rare, morfologicamente eterogenee o semplicemente perché non è pensabile raccogliere ground truth
per ogni possibile anomalia. L'*anomaly detection* non supervisionata affronta il problema in modo
diverso: il modello apprende esclusivamente la distribuzione del tessuto **sano**, e qualunque
scostamento significativo da tale distribuzione viene trattato come potenziale anomalia.

Questo progetto confronta, a parità di dataset e protocollo sperimentale, **tre paradigmi**
di anomaly detection non supervisionata applicati a slice MRI cerebrali:

1. **Isolation Forest** — baseline classica di machine learning su pixel grezzi
2. **CNN Autoencoder** — approccio generativo/ricostruttivo (con quattro varianti: loss ablation,
   denoising gaussiano, denoising strutturale coarse, post-processing morfologico)
3. **PatchCore** — transfer learning da rete pre-addestrata su ImageNet, memory bank + KNN
   (Roth et al., 2022)

L'obiettivo non è solo individuare l'immagine anomala (**image-level detection**), ma anche
localizzare la lesione a livello di singolo pixel (**pixel-level segmentation**), usando le
maschere di segmentazione **esclusivamente in fase di valutazione finale**, mai durante
l'addestramento o la selezione del modello.

---

## 2. Dataset

Il dataset utilizzato è **BraTS2021** (Brain Tumor Segmentation Challenge 2021), riorganizzato
secondo il formato del framework **MedIAnomaly**, che garantisce riproducibilità scientifica e
un confronto equo tra i diversi paradigmi.

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
- **Controllo data leakage:** nessuna sovrapposizione di pazienti tra `train` e i due split di
  `test` (`TRAIN ∩ TEST NORMAL = 0`, `TRAIN ∩ TEST TUMOR = 0`). Una sovrapposizione di 174 pazienti
  tra `TEST NORMAL` e `TEST TUMOR` è invece attesa: sono pazienti oncologici con slice sane in
  alcune sezioni assiali e slice tumorali in altre.

### 2.3 Slice per paziente

| Split | Min | Max | Media | Mediana | Std |
|---|---:|---:|---:|---:|---:|
| Train | 1 | 14 | 4.51 | 4.00 | 2.69 |
| Test normal | 1 | 14 | 4.73 | 4.00 | 2.87 |
| Test tumor | 1 | 14 | 9.79 | 10.00 | 3.12 |

### 2.4 Statistiche di intensità e sovrapposizione delle distribuzioni

L'analisi esplorativa (`src/data_analysis/eda.py`, output completo in
[`results/eda/eda_report.txt`](results/eda/eda_report.txt)) evidenzia una significativa
sovrapposizione tra intensità sane e tumorali, motivando la difficoltà del problema:

| Metrica | Normal | Tumor |
|---|---:|---:|
| Mean intensity | 30.25 | 36.08 |
| Std intensity | 45.70 | 51.61 |
| p95 intensity | 113.91 | 128.18 |
| Nonzero fraction | 0.330 | 0.356 |

- **Distanza di Wasserstein (globale):** 5.77
- **Distanza di Wasserstein (pixel non-zero):** 12.39
- **Area media della lesione:** 4.46% dei pixel per slice (mediana 4.03%, range 0.35%–14.89%)
- **PCA esplorativa (2 componenti, campione bilanciato N=1500):** varianza spiegata cumulativa
  ≈ 44.3% (PC1 = 30.7%, PC2 = 13.7%) — le due classi non sono linearmente separabili nello spazio
  dei pixel grezzi, giustificando l'adozione di rappresentazioni apprese (feature learning).

Le figure generate (distribuzione delle classi, istogrammi di intensità, immagini medie
sano/tumore, mappa di frequenza spaziale delle lesioni, esempi visivi, overlay FLAIR+segmentazione,
proiezione PCA) sono salvate in `results/eda/figures/`.

---

## 3. Protocollo sperimentale

Per garantire un confronto rigoroso e privo di *data leakage*, tutti i modelli seguono lo stesso
protocollo:

- **Training:** esclusivamente immagini sane (`train/`)
- **Validation** (15% delle immagini sane di train, split fisso con seed 42): usata per
  *early stopping*, model selection e calibrazione delle soglie — **mai** il test set
- **Test:** sano + tumorale, usato **una sola volta** per la valutazione finale
- **Soglia image-level:** 95° percentile degli score di anomalia calcolato sul validation set sano
- **Soglia pixel-level:** 99° percentile delle anomaly map calcolato sul validation set sano
- Nessun tuning di soglia sul test set, nessuna informazione tumorale (label o maschera) utilizzata
  in training o validation
- Seed fisso (`SEED = 42`) per la riproducibilità di split, inizializzazione dei pesi e sampling

---

## 4. Modelli implementati

### 4.1 Isolation Forest — baseline classica

Modello di ensemble basato sull'isolamento casuale dei punti in uno spazio ad alta dimensionalità.
Le immagini vengono appiattite (`64×64 → 4096`) e standardizzate (`StandardScaler` fittato solo
sul train). `contamination="auto"`, soglia decisionale nativa a 0, nessun tuning.
Rappresenta il limite del ML classico: efficienza computazionale estrema ma nessuna capacità di
localizzazione spaziale (perdita totale della struttura 2D con il flattening).

### 4.2 CNN Autoencoder — approccio generativo/ricostruttivo

Autoencoder convoluzionale fully-convolutional (nessun bottleneck lineare), addestrato a
ricostruire immagini sane. Lo score di anomalia a livello immagine è l'errore quadratico medio di
ricostruzione (MSE); la mappa di anomalia pixel-level è l'errore assoluto `|input - reconstruction|`.

Il modello è stato sviluppato per iterazioni successive (`src/dl/cnn_ae/`):

| Script | Variante | Descrizione |
|---|---|---|
| `0.baseline.py` | Baseline (MSE) | Encoder/decoder con `BatchNorm` + `ReLU`, bottleneck lineare (`fc`), loss MSE pura |
| `1.loss_ablation.py` | Ablation loss , confronto tra MSE / L1 / MSE+L1 |
| `2.denoising.py` | + Denoising | Rumore gaussiano *coarse* (griglia 16×16 interpolata) iniettato in input come regolarizzazione (denoising autoencoder) |
| `3.post_processing.py` | + Post-processing | Post-processing morfologico sulle anomaly map (rimozione componenti connesse piccole, `binary_closing`) |


### 4.3 PatchCore — transfer learning + memory bank

Basato su [Roth et al., CVPR 2022](https://arxiv.org/abs/2106.08265). Nessun training è eseguito:
si sfrutta una ResNet-18 **pre-addestrata su ImageNet** (l'input single-channel viene replicato
su 3 canali) per estrarre feature locali gerarchiche da `layer1` e `layer2`. Le feature estratte
dalle immagini sane di train formano una **Memory Bank**, compressa tramite coreset subsampling
casuale (5%). In fase di test, lo score di anomalia per ogni patch è la distanza euclidea (KNN,
K=1) dalla patch più vicina in memoria; la mappa risultante viene smussata con un filtro
gaussiano (`sigma=2.0`).

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
│   │   │   ├── 0.baseline.py        # Autoencoder con loss MSE (bottleneck lineare)
│   │   │   ├── 1.loss_ablation.py   # Confronto MSE / L1 / MSE+L1
│   │   │   ├── 2.denoising.py       # + rumore gaussiano coarse in input
│   │   │   ├── 3.post_processing.py # + post-processing morfologico sulle mappe
│   │   │
│   │   │
│   │   └── PatchCore/
│   │       └── 1.baseline.py        # memory bank + KNN su feature pre-addestrate ImageNet
│   │
│   ├── collect_results.py           # Aggregazione automatica delle metriche di tutti i modelli
│   ├── final_plot.py                # Grafici comparativi finali fra i 4 modelli
│   ├── patient_level.py             # Aggregazione slice-level → patient-level (regola del max score)
│   └── confidence.py                # Intervalli di confidenza bootstrap (95%, N=2000) + forest plot
│
├── results/                          # Output di ciascun esperimento (metriche, report, figure)
│   ├── eda/                          # Report e figure dell'analisi esplorativa
|   ├──dataloader_validation          
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

Tutti i valori sono estratti automaticamente da `src/collect_results.py` a partire dai CSV di
ciascun esperimento (`results/summary/model_comparison.csv`). Per ogni modello viene riportata
la variante di riferimento più significativa.

### 6.1 Confronto principale — image-level e pixel-level

| Modello | Variante | Img AUROC | Img AP | Img F1 | Img Sens. | Img Spec. | Pixel AUROC | Pixel Dice | Tempo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | baseline | 0.6704 | 0.7935 | 0.4892 | 0.3475 | 0.8273 | — | — | **0.3 s** |
| CNN Autoencoder | MSE+L1 + denoising + post-processing | 0.7994 | 0.9021 | 0.5469 | 0.3860 | 0.9396 | 0.9038 | **0.3768** | 820 s |
| **PatchCore** | final | **0.9037** | **0.9590** | **0.7751** | **0.6458** | **0.9517** | **0.9561** | 0.3127 | 850 s |




### 6.2 Intervalli di confidenza bootstrap (95%, N=2.000 resample)

| Modello | AUROC | IC 95% | AP | IC 95% |
|---|---:|---|---:|---|
| Isolation Forest | 0.6704 | [0.648, 0.692] | 0.7935 | [0.773, 0.816] |
| CNN Autoencoder | 0.6723 | [0.651, 0.693] | 0.8187 | [0.800, 0.838] |
| **PatchCore** | **0.9037** | [0.892, 0.914] | **0.9590** | [0.953, 0.965] |

Gli intervalli di confidenza di PatchCore non si sovrappongono con quelli di nessun altro modello,
a supporto di una superiorità robusta (non solo puntuale) rispetto alle altre tre architetture.
Script: `src/confidence.py` → `results/summary/bootstrap_analysis/`.

### 6.3 Valutazione patient-level

Oltre alla valutazione slice-level, `src/patient_level.py` aggrega gli score alla granularità del
paziente secondo la regola clinica `patient_score = max(anomaly_score)` sulle slice del paziente
(`patient_label = 1` se almeno una slice è tumorale), ricalcolando AUROC e AP a livello paziente
per tutti i modelli compatibili. Risultati numerici e grafici (barplot slice- vs patient-level,
curve ROC patient-level) in `results/summary/patient_level/`.

### 6.4 Grafici comparativi finali

Generati da `src/final_plot.py` in `results/summary/final_comparison/`:
1. Confronto image-level (AUROC / AP)
2. Confronto pixel-level (AUROC / Dice)
3. Scatter plot Detection vs. Localization trade-off
4. Metriche alla soglia operativa (Sensitivity / Specificity / F1)
5. Costo computazionale (scala logaritmica)

---

## 7. Analisi critica

- **PatchCore** ottiene le prestazioni migliori sia in detection sia in localizzazione, grazie a
  feature pre-addestrate su ImageNet molto più informative dei pixel grezzi o di feature apprese
  da zero su un dataset di dimensioni ridotte (4.211 immagini di train). Il vantaggio è
  statisticamente robusto (bootstrap CI non sovrapposti, §6.3).
- Il **CNN Autoencoder** mostra un buon Pixel AUROC (0.92) nella variante finale, ma un Dice più
  contenuto (0.48): la soglia P99 selezionata su validation potrebbe non essere ottimale per la
  metrica Dice; un'analisi threshold-sweep (vedi §11) potrebbe migliorare ulteriormente la
  localizzazione.
- **Isolation Forest**, pur essendo la baseline più semplice ed estremamente efficiente (< 1 s di
  training), rimane competitiva in detection (Img AUROC 0.67) grazie alla forte separabilità
  parziale nello spazio dei pixel grezzi, ma non è per costruzione in grado di produrre alcuna
  mappa di localizzazione spaziale.

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

> Per l'addestramento dei modelli deep (CNN Autoencoder, CutPaste, PatchCore) è fortemente
> consigliata una GPU CUDA; tutti gli script rilevano automaticamente il device disponibile
> (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

---

## 9. Come riprodurre gli esperimenti

```bash
# 1. Analisi esplorativa del dataset
python src/data_analysis/eda.py

# 2. Dataloader e sua validazione
python src/data_analysis/dataloader.py
python src/data_analysis/validate_dataloader.py

# 3. Baseline Isolation Forest
python src/ml/ml_baseline.py

# 4. CNN Autoencoder (in ordine progressivo — ogni script è indipendente)
python src/dl/cnn_ae/0.baseline.py
python src/dl/cnn_ae/1.loss_ablation.py
python src/dl/cnn_ae/2.denoising.py
python src/dl/cnn_ae/3.post_processing.py


# 5. PatchCore
python src/dl/PatchCore/1.baseline.py

# 6. Aggregazione risultati e grafici comparativi finali
python src/collect_results.py
python src/final_plot.py

# 7. Estensioni statistiche e cliniche
python src/confidence.py       # intervalli di confidenza bootstrap + forest plot
python src/patient_level.py    # valutazione a livello di paziente
```

Ogni script scrive i propri output (report `.txt`, CSV con le metriche, figure `.png`) in una
sottocartella dedicata di `results/`, così da poter rieseguire un singolo esperimento senza
impattare gli altri.

---

## 10. Limitazioni note

- **Elaborazione 2D slice-per-slice:** il volume MRI 3D viene trattato come una sequenza
  indipendente di slice assiali, con conseguente perdita della continuità volumetrica lungo
  l'asse Z.
- **Downsampling a 64×64:** riduce il costo computazionale ma comprime dettagli fini della
  lesione, con possibile impatto sulla localizzazione pixel-level (soprattutto per lesioni
  piccole, minoranza nella distribuzione delle aree tumorali osservata in EDA).
- **Sensibilità della soglia P99 (pixel-level):** calibrata sulla distribuzione dell'errore sano
  in validation; non necessariamente ottimale rispetto a metriche come il Dice score.
- **Assenza di test di significatività pairwise formalizzato:** sono disponibili intervalli di
  confidenza bootstrap per-modello (§6.3), ma non un test statistico pairwise esplicito
  (es. bootstrap sulla differenza di AUROC o DeLong test) tra coppie di modelli.


---

## 11. Sviluppi futuri

- Estensione a modelli **3D** (3D-CNN, 3D Autoencoder) per sfruttare la continuità volumetrica
  tra slice adiacenti.
- Adozione di **modelli generativi di frontiera** (Diffusion Models, Masked Autoencoders) come
  paradigma ricostruttivo alternativo al CNN Autoencoder classico.
- **Threshold-sweep** sistematico sulla soglia pixel-level per ottimizzare esplicitamente il
  Dice score, invece di fissare a priori il percentile P99.
- Implementazione di un **test di significatività statistica pairwise** (es. bootstrap sulla
  differenza di AUROC tra coppie di modelli, o DeLong test) a completamento degli intervalli di
  confidenza già calcolati.
- Validazione su **coorti esterne** (altri dataset BraTS o altre patologie oncologiche
  cerebrali) per verificare la generalizzazione dei risultati.

---
