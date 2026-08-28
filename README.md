# Anomaly Detection in the Healthcare Field
### Rilevamento non supervisionato di tumori cerebrali su risonanze magnetiche (BraTS2021)

Studio comparativo di tecniche di *anomaly detection* per l'identificazione e la
localizzazione di tumori cerebrali a partire da slice MRI, addestrate **esclusivamente su immagini sane**.

---

## 1. Obiettivo

Confrontare, a parità di dataset e protocollo sperimentale, quattro paradigmi di anomaly detection
non supervisionata applicati alla diagnosi per immagini:

1. **Isolation Forest** — baseline classica di machine learning su pixel grezzi
2. **CNN Autoencoder** — approccio generativo/ricostruttivo (con varianti: loss ablation, denoising, post-processing)
3. **CutPaste** — apprendimento self-supervised tramite task pretestuale
4. **PatchCore** — transfer learning da rete pre-addestrata su ImageNet (memory bank + KNN)

L'obiettivo non è solo individuare l'immagine anomala (**image-level**), ma anche localizzare la
lesione a livello di pixel (**pixel-level**), usando le maschere di segmentazione solo in fase di
valutazione finale.

---

## 2. Dataset

**BraTS2021**, riorganizzato nel formato (stile MedIAnomaly):

```
BraTS2021/
├── train/                  # esclusivamente slice sane (label = 0)
└── test/
    ├── normal/             # slice sane (label = 0)
    ├── tumor/               # slice con tumore (label = 1)
    └── annotation/          # maschere di segmentazione (solo per le slice tumor)
```

Caratteristiche principali (da `results/eda/eda_report.txt`):

| Split | Immagini | Pazienti |
|---|---|---|
| Train | 4211 | 933 |
| Test normal | 828 | 175 |
| Test tumor | 1948 | 199 |

- Risoluzione immagini: 208×208 originali, ridimensionate a **64×64** per l'addestramento
- Canali: 1 (scala di grigi), range **[0, 1]**
- Nessuna sovrapposizione di pazienti tra train e test (leakage check superato)
- Area media della lesione: ~4.5% dei pixel per slice

---

## 3. Protocollo sperimentale

- **Training**: solo immagini sane (`train/`)
- **Validation** (15% delle immagini sane di train): usata per *early stopping*, model selection e
  calibrazione delle soglie — mai il test set
- **Test**: sano + tumorale, usato **una sola volta** per la valutazione finale
- **Soglia image-level**: 95° percentile degli score di anomalia sul validation set sano
- **Soglia pixel-level**: 99° percentile delle anomaly map sul validation set sano
- Nessun tuning di soglia sul test set, nessuna informazione tumorale usata in training/validation
- Seed fisso (42) per la riproducibilità

---

## 4. Struttura del repository

```
src/
├── data_analysis/
│   ├── dataloader.py            # Dataset PyTorch (BraTSDataset)
│   ├── eda.py                   # Analisi esplorativa completa + figure
│   ├── inspect_brats.py         # Ispezione strutturale del dataset grezzo
│   └── validate_dataloader.py   # Validazione automatica del dataloader
│
├── ml/
│   └── ml_baseline.py           # Isolation Forest (baseline ML pura)
│
├── dl/
│   ├── cnn_ae/
│   │   ├── 0.baseline.py        # Autoencoder con loss MSE
│   │   ├── 1.loss_ablation.py   # Confronto MSE / L1 / MSE+L1
│   │   ├── 2.denoising.py       # + rumore gaussiano coarse in input
│   │   └── 3.post_processing.py # + post-processing morfologico sulle mappe
│   │
│   ├── CutPaste/
│   │   ├── 1.baseline.py        # training self-supervised + valutazione image-level
│   │   └── 1.pixel-level.py     # estensione pixel-level (feature spaziali ResNet-18)
│   │
│   └── PatchCore/
│       └── 1.baseline.py        # memory bank + KNN su feature pre-addestrate
│
└── final_plot.py                # grafici comparativi finali fra i 4 modelli

results/                          # output di ciascun esperimento (metriche, report, figure)
```

---

## 5. Risultati principali

| Modello | Image AUROC | Image AP | Pixel AUROC | Dice |
|---|---|---|---|---|
| Isolation Forest | 0.6704 | 0.7935 | — | — |
| CNN Autoencoder (MSE+L1, denoising) | 0.6723 | 0.8187 | 0.9189 | 0.4778 |
| CutPaste | 0.6178 | 0.7602 | 0.7241 | 0.0524 |
| **PatchCore** | **0.9037** | **0.9590** | **0.9561** | 0.3127 |

**Osservazioni:**
- **PatchCore** ottiene le migliori prestazioni sia in detection che in localizzazione, grazie a
  feature pre-addestrate su ImageNet molto più informative dei pixel grezzi o di feature apprese
  da zero su un dataset di dimensioni ridotte.
- **CutPaste** è il modello con le prestazioni peggiori, anche sotto la baseline Isolation Forest.
  Questo è plausibilmente dovuto al fatto che il task pretestuale (patch "incollata" localmente)
  è pensato per anomalie di texture in ambito industriale, mentre un tumore cerebrale altera anche
  la struttura anatomica globale della slice.
- Il **CNN Autoencoder** mostra un buon Pixel AUROC (0.92) ma un Dice più contenuto (0.48): la
  soglia P99 potrebbe non essere ottimale per la metrica Dice (si consiglia un'analisi threshold-sweep,
  vedi sezione Limitazioni).

Grafici comparativi completi in `results/final_comparison/`.

---

## 6. Come riprodurre gli esperimenti

### Requisiti

```
python >= 3.10
torch, torchvision
scikit-learn
numpy, pandas
matplotlib, seaborn
scipy
pillow
joblib
```

### Esecuzione

```bash
# 1. Analisi esplorativa del dataset
python src/data_analysis/eda.py

# 2. Dataloader e la sua validazione
python src/data_analysis/dataloader.py
python src/data_analysis/validate_dataloader.py

# 3. Baseline Isolation Forest
python src/ml/ml_baseline.py

# 4. CNN Autoencoder (in ordine progressivo)
python src/dl/cnn_ae/0.baseline.py
python src/dl/cnn_ae/1.loss_ablation.py
python src/dl/cnn_ae/2.denoising.py
python src/dl/cnn_ae/3.post_processing.py

# 5. CutPaste
python src/dl/CutPaste/1.baseline.py
python src/dl/CutPaste/1.pixel-level.py   # richiede il modello salvato dal punto precedente

# 6. PatchCore
python src/dl/PatchCore/1.baseline.py

# 7. Grafici comparativi finali
python src/final_plot.py
---
