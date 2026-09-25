# Anomaly Detection in the Healthcare Field

### Rilevamento e localizzazione non supervisionata di tumori cerebrali su risonanze magnetiche (BraTS2021)

Studio comparativo di tre paradigmi di *anomaly detection* — Machine Learning classico, Deep Learning ricostruttivo e Transfer Learning — per l'identificazione e la localizzazione di tumori cerebrali a partire da slice MRI, con modelli addestrati esclusivamente su immagini sane.

---

## 1. Obiettivo e motivazioni

Nella diagnostica per immagini, gli approcci di classificazione supervisionata richiedono generalmente una quantità significativa di dati annotati per le diverse condizioni patologiche considerate. Questo requisito può risultare particolarmente oneroso nel caso di patologie rare, morfologicamente eterogenee o caratterizzate da una elevata variabilità nella presentazione radiologica.

L'*anomaly detection* affronta il problema da una prospettiva differente: il modello viene costruito utilizzando esclusivamente dati rappresentativi della condizione normale e gli scostamenti rispetto alla distribuzione appresa vengono trattati come potenziali anomalie.

Il progetto confronta, a parità di dataset e protocollo sperimentale, tre differenti paradigmi di *anomaly detection* applicati a slice MRI cerebrali:

1. **Isolation Forest** — baseline di Machine Learning classico operante sui pixel grezzi;
2. **CNN Autoencoder** — approccio Deep Learning ricostruttivo basato sulla ricostruzione di immagini sane;
3. **PatchCore** — approccio basato su Transfer Learning e confronto tra rappresentazioni locali tramite una *Memory Bank*.

L'obiettivo è valutare i tre approcci sia per la capacità di identificare immagini anomale (*image-level detection*) sia per la capacità di localizzare spazialmente le regioni patologiche (*pixel-level localization*).

Le maschere di segmentazione delle lesioni vengono utilizzate esclusivamente nella fase di valutazione finale e non intervengono durante il training, la selezione del modello o la calibrazione delle soglie.

---

## 2. Dataset

Il dataset utilizzato è **BraTS2021** (*Brain Tumor Segmentation Challenge 2021*), riorganizzato secondo una struttura compatibile con il framework MedIAnomaly.

La riorganizzazione consente di adottare una suddivisione standardizzata dei dati e un protocollo coerente tra i diversi approcci di anomaly detection.

### 2.1 Struttura su disco

```text
BraTS2021/

├── train/                  # esclusivamente slice sane (label = 0)
│
└── test/
    ├── normal/             # slice sane (label = 0)
    ├── tumor/              # slice con tumore (label = 1)
    └── annotation/         # maschere di segmentazione delle slice tumor
```

> **Nota:** la cartella `BraTS2021/` non è inclusa nel repository per motivi di dimensione e di licenza dei dati. Il dataset deve essere scaricato separatamente e posizionato nella root del progetto.

### 2.2 Composizione dei dati

| Split         | Immagini | Pazienti |
| ------------- | -------: | -------: |
| Train (sano)  |    4.211 |      933 |
| Test — normal |      828 |      175 |
| Test — tumor  |    1.948 |      199 |
| Annotation    |    1.948 |      199 |

Caratteristiche principali:

* **Risoluzione originale:** 208 × 208 pixel;
* **Risoluzione di lavoro:** 64 × 64 pixel, ottenuta durante il caricamento tramite il dataloader;
* **Canali:** 1, in scala di grigi;
* **Range dei valori:** conversione nell'intervallo `[0, 1]` tramite `ToTensor()`;
* **Normalizzazione successiva:** non viene applicata alcuna ulteriore normalizzazione statistica;
* **Controllo del data leakage:** non sono presenti pazienti condivisi tra `train` e `test normal` né tra `train` e `test tumor`.

È invece prevista una sovrapposizione tra `test/normal` e `test/tumor`: 174 pazienti compaiono in entrambi gli split, poiché uno stesso paziente può presentare slice sane in alcune sezioni assiali e slice tumorali in altre. Questa caratteristica viene considerata esplicitamente nella successiva valutazione *patient-level*.

### 2.3 Slice per paziente

| Split       | Min | Max | Media | Mediana |  Std |
| ----------- | --: | --: | ----: | ------: | ---: |
| Train       |   1 |  14 |  4.51 |    4.00 | 2.69 |
| Test normal |   1 |  14 |  4.73 |    4.00 | 2.87 |
| Test tumor  |   1 |  14 |  9.79 |   10.00 | 3.12 |

La diversa numerosità media di slice per paziente tra gli split viene tenuta in considerazione nella successiva aggregazione a livello paziente.

### 2.4 Statistiche di intensità e analisi esplorativa

L'analisi esplorativa implementata in `src/data_analysis/eda.py` evidenzia una significativa sovrapposizione tra le distribuzioni di intensità delle immagini normali e tumorali:

| Metrica          | Normal |  Tumor |
| ---------------- | -----: | -----: |
| Mean intensity   |  30.25 |  36.08 |
| Std intensity    |  45.70 |  51.61 |
| p95 intensity    | 113.91 | 128.18 |
| Nonzero fraction |  0.330 |  0.356 |

Ulteriori indicatori:

* **Distanza di Wasserstein globale:** 5.77;
* **Distanza di Wasserstein sui pixel non-zero:** 12.39;
* **Area media della lesione:** 4.46% dei pixel per slice;
* **Area mediana della lesione:** 4.03%;
* **Range dell'area tumorale:** 0.35%–14.89%.

È stata inoltre effettuata un'analisi PCA esplorativa sulle immagini rappresentate nello spazio dei pixel grezzi. Considerando due componenti principali e un campione bilanciato di 1.500 immagini, la varianza spiegata cumulativa è pari a circa il 44.3%:

* PC1: 30.7%;
* PC2: 13.7%;
* varianza cumulativa: circa 44.3%.

La proiezione sulle prime due componenti mostra una significativa sovrapposizione tra le due classi. Tale osservazione motiva l'analisi di rappresentazioni apprese oltre alla semplice elaborazione dei pixel grezzi.

---

## 3. Protocollo sperimentale

Per garantire un confronto coerente tra i modelli ed evitare *data leakage*, tutti gli esperimenti seguono lo stesso schema generale.

### Training

Il training utilizza esclusivamente le immagini sane presenti nella cartella `train/`.

### Validation

Il 15% delle immagini sane di training viene utilizzato come *validation set*, mediante uno split fisso con `SEED = 42`.

Il validation set viene utilizzato per:

* *early stopping*, ove applicabile;
* selezione del modello;
* calibrazione delle soglie operative.

Nessuna informazione proveniente dalle immagini tumorali o dalle relative maschere viene utilizzata durante questa fase.

### Test

Il test set contiene sia immagini normali sia immagini tumorali e viene utilizzato esclusivamente per la valutazione finale delle prestazioni.

Le maschere di segmentazione vengono utilizzate solamente per il calcolo delle metriche *pixel-level*.

### Soglie image-level

Per CNN Autoencoder e PatchCore, la soglia operativa *image-level* viene calibrata sul validation set sano utilizzando il 95-esimo percentile della distribuzione degli *anomaly score*.

Isolation Forest utilizza invece la propria soglia decisionale nativa, corrispondente a `decision_function = 0`, senza tuning della soglia sul test set.

### Soglie pixel-level

Per CNN Autoencoder e PatchCore, la soglia pixel-level viene calibrata sul validation set sano utilizzando il 99-esimo percentile dei valori delle *anomaly map*.

Anche in questo caso, nessuna informazione relativa alle lesioni del test set viene utilizzata per determinare la soglia.

### Riproducibilità

Il progetto utilizza `SEED = 42` per rendere riproducibili:

* lo split train/validation;
* l'inizializzazione dei pesi dei modelli che prevedono training;
* le procedure di sampling;
* le operazioni casuali utilizzate negli esperimenti.

---

## 4. Modelli implementati

### 4.1 Isolation Forest — baseline di Machine Learning

Isolation Forest è utilizzato come baseline di Machine Learning classico.

Le immagini vengono ridimensionate a 64 × 64 pixel e successivamente appiattite in un vettore di 4.096 caratteristiche:

```text
64 × 64 → 4096
```

I vettori vengono standardizzati mediante `StandardScaler`, fittato esclusivamente sui dati di training.

Il modello utilizza:

```text
contamination = "auto"
```

e la soglia decisionale nativa del modello.

L'approccio costituisce una baseline a basso costo computazionale, ma non conserva esplicitamente la struttura spaziale bidimensionale dell'immagine dopo il flattening. Di conseguenza, non produce una mappa spaziale dell'anomalia e viene valutato esclusivamente a livello *image-level*.

### 4.2 CNN Autoencoder — approccio ricostruttivo

Il CNN Autoencoder viene addestrato esclusivamente sulle immagini sane con l'obiettivo di ricostruire la distribuzione della normalità.

L'architettura utilizza:

* convoluzioni per l'estrazione della rappresentazione;
* processo di encoding/decoding;
* loss ibrida MSE + L1;
* denoising gaussiano *coarse* durante il training.

Lo *image-level anomaly score* è calcolato come errore quadratico medio di ricostruzione:

```text
MSE(input, reconstruction)
```

La *anomaly map* pixel-level è invece ottenuta mediante errore assoluto:

```text
|input - reconstruction|
```

La mappa viene successivamente sottoposta alla soglia pixel-level e alle operazioni di post-processing previste dalla pipeline.

### 4.3 PatchCore — Transfer Learning e Memory Bank

PatchCore segue l'approccio proposto da Roth et al. (CVPR 2022).

Il modello non esegue il fine-tuning della rete backbone. Viene utilizzata una **ResNet-18 pre-addestrata su ImageNet** per estrarre rappresentazioni locali dalle immagini.

Poiché le immagini MRI sono monocanale, l'input viene replicato sui tre canali richiesti dalla rete pre-addestrata.

Le feature vengono estratte da:

* `layer1`;
* `layer2`.

Le rappresentazioni locali ottenute dalle immagini sane del training set vengono utilizzate per costruire una **Memory Bank**.

La Memory Bank viene successivamente ridotta mediante *coreset subsampling* casuale con rapporto del 5%.

Durante l'inferenza, per ciascuna patch viene calcolata la distanza euclidea dalla patch più vicina presente nella Memory Bank utilizzando KNN con:

```text
K = 1
```

Le distanze vengono quindi aggregate per ottenere la mappa spaziale dell'anomalia. La mappa viene infine sottoposta a smoothing mediante filtro gaussiano con:

```text
sigma = 2.0
```

Il metodo non richiede un training supervisionato sul dataset MRI e non utilizza le maschere tumorali nella costruzione della Memory Bank.

---

## 5. Struttura del repository

```text
├── src/
│   ├── data_analysis/
│   │   ├── dataloader.py
│   │   ├── eda.py
│   │   └── validate_dataloader.py
│   │
│   ├── ml/
│   │   └── ml_baseline.py
│   │
│   ├── dl/
│   │   ├── cnn_ae/
│   │   │   └── ae_baseline.py
│   │   │
│   │   └── PatchCore/
│   │       └── pc_baseline.py
│   │
│   ├── collect_results.py
│   ├── final_plot.py
│   ├── compare_heatmaps.py
│   ├── patient_level.py
│   └── confidence.py
│
├── results/
│   ├── eda/
│   ├── ml_baseline/
│   ├── cnn_autoencoder_/
│   ├── patchcore/
│   └── summary/
│
├── README.md
└── .gitignore
```


## 6. Risultati

Tutti i valori riportati nei risultati vengono estratti automaticamente mediante `src/collect_results.py` a partire dai file CSV prodotti dai singoli esperimenti.

### 6.1 Confronto principale

| Modello          |  Img AUROC |     Img AP |     Img F1 |  Img Sens. |  Img Spec. | Pixel AUROC | Pixel Dice |     Tempo |
| ---------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ----------: | ---------: | --------: |
| Isolation Forest |     0.6704 |     0.7935 |     0.4892 |     0.3475 |     0.8273 |           — |          — | **1.4 s** |
| CNN Autoencoder  |     0.7994 |     0.9021 |     0.5469 |     0.3860 |     0.9396 |      0.9086 |     0.3753 | 2098.23 s |
| **PatchCore**    | **0.9037** | **0.9590** | **0.7751** | **0.6458** | **0.9517** |  **0.9561** | **0.4414** |  377.69 s |

I risultati mostrano differenze tra i tre approcci sia in termini di capacità discriminativa a livello immagine sia, per i modelli che producono mappe spaziali, in termini di localizzazione pixel-level.

Isolation Forest non produce una rappresentazione spaziale dell'anomalia e non viene pertanto valutato mediante metriche pixel-level.

PatchCore presenta i valori più elevati tra i modelli considerati nelle principali metriche image-level e pixel-level riportate nella tabella.

### 6.2 Validazione statistica

La significatività delle differenze tra gli AUROC viene analizzata mediante:

* bootstrap con 2.000 resampling;
* intervalli di confidenza al 95%;
* test di DeLong per confronti pairwise tra curve ROC.

Il test di DeLong rileva una differenza statisticamente significativa dell'AUROC tra PatchCore e Isolation Forest:

```text
ΔAUROC = +0.233
p < 0.001
```

e tra PatchCore e CNN Autoencoder:

```text
ΔAUROC = +0.104
p < 0.001
```

Questi risultati indicano che le differenze osservate negli AUROC di PatchCore rispetto agli altri due modelli sono statisticamente significative nel campione di test considerato.

Gli intervalli di confidenza bootstrap vengono inoltre utilizzati per quantificare l'incertezza associata alle stime delle metriche.

### 6.3 Valutazione patient-level

Oltre alla valutazione slice-level, il progetto considera una valutazione a livello di paziente.

Lo script:

```text
src/patient_level.py
```

aggrega gli *anomaly score* delle slice appartenenti allo stesso paziente mediante la regola:

```text
patient_score = max(anomaly_score)
```

Il paziente viene considerato positivo se almeno una delle sue slice è associata alla classe tumorale:

```text
patient_label = 1
```

In caso contrario:

```text
patient_label = 0
```

La procedura consente di passare dalla valutazione della singola slice alla valutazione della capacità del modello di identificare pazienti che presentano almeno una regione tumorale.

I risultati numerici e i grafici della valutazione patient-level sono salvati in:

```text
results/summary/patient_level/
```

### 6.4 Grafici comparativi

Gli script di aggregazione generano automaticamente diversi grafici nella directory:

```text
results/summary/
```

Tra questi:

```text
image_level_performance.png
pixel_level_performance.png
slice_level_roc_curves.png
slice_level_pr_curves.png
pixel_dice_boxplot.png
detection_localization_tradeoff.png
training_computational_cost.png
```

I grafici consentono di confrontare:

* metriche image-level;
* metriche pixel-level;
* curve ROC;
* curve Precision–Recall;
* distribuzione del Dice per slice;
* relazione tra detection e localization;
* costo computazionale dei diversi approcci.

---

## 7. Analisi critica

I risultati evidenziano differenze sostanziali tra i tre paradigmi analizzati.

**PatchCore** presenta i valori più elevati tra i modelli confrontati nelle principali metriche image-level considerate e mostra inoltre prestazioni pixel-level superiori a quelle del CNN Autoencoder nelle metriche riportate. Il comportamento è compatibile con l'utilizzo di rappresentazioni locali pre-addestrate e con il confronto delle feature mediante una Memory Bank.

**CNN Autoencoder** consente sia la classificazione image-level sia la produzione di mappe di anomalia spaziali. Le prestazioni pixel-level mostrano tuttavia una differenza tra la capacità discriminativa dei valori continui della anomaly map e la qualità della maschera binaria ottenuta dopo sogliatura e post-processing. Tale comportamento è compatibile con la possibilità che un autoencoder addestrato esclusivamente su immagini normali ricostruisca parzialmente anche caratteristiche patologiche.

**Isolation Forest** costituisce una baseline a costo computazionale molto contenuto. Il modello opera sui pixel appiattiti e non produce una rappresentazione spaziale dell'anomalia, limitando la valutazione alla detection image-level.

I tempi di esecuzione devono essere interpretati in relazione all'hardware e all'ambiente software utilizzati durante gli esperimenti. Essi consentono quindi un confronto relativo tra gli approcci nell'ambito della configurazione sperimentale adottata, ma non costituiscono benchmark assoluti indipendenti dall'hardware.

---

## 8. Ambiente software e hardware

### Software

Il progetto è stato sviluppato in Python 3.10 o versione successiva.

Principali dipendenze:

```text
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

### Hardware

Gli esperimenti Deep Learning possono essere eseguiti su GPU CUDA.

Il dispositivo viene selezionato automaticamente tramite:

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## 9. Installazione

Si consiglia l'utilizzo di un ambiente virtuale.

```bash
python -m venv venv
source venv/bin/activate
venv\Scripts\activate
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn scipy pillow joblib
```

Per l'esecuzione degli esperimenti CNN Autoencoder e PatchCore è consigliata una GPU CUDA, soprattutto per ridurre i tempi di elaborazione.

---

## 10. Come riprodurre gli esperimenti

```bash
python src/data_analysis/eda.py

python src/data_analysis/dataloader.py
python src/data_analysis/validate_dataloader.py

python src/ml/ml_baseline.py
python src/dl/cnn_ae/ae_baseline.py
python src/dl/PatchCore/pc_baseline.py

python src/collect_results.py
python src/final_plot.py

python src/confidence.py
python src/patient_level.py
python src/compare_heatmaps.py
```

Ogni script salva i propri output nella relativa sottocartella di `results/`.

---

## 11. Limitazioni note

### Elaborazione 2D slice-by-slice

Il volume MRI 3D viene trattato come una sequenza di slice assiali indipendenti. Questa scelta semplifica il problema e riduce il costo computazionale, ma comporta la perdita delle informazioni di continuità anatomica lungo l'asse Z.

### Downsampling a 64 × 64 pixel

Le immagini vengono ridimensionate dalla risoluzione originale di 208 × 208 alla risoluzione di lavoro di 64 × 64 pixel.

Il downsampling riduce il costo computazionale ma può comportare una perdita di dettagli fini della lesione, con possibile impatto sulla localizzazione pixel-level, soprattutto nel caso di regioni patologiche di piccola estensione.

### Sensibilità della soglia pixel-level

La soglia pixel-level viene calibrata mediante il 99-esimo percentile della distribuzione degli errori spaziali osservati sul validation set sano.

Questa scelta consente di definire una soglia senza utilizzare informazioni tumorali, ma non è necessariamente ottimale rispetto a metriche di segmentazione supervisionate come il Dice score.

### Post-processing delle anomaly map

La maschera binaria ottenuta dalla sogliatura viene sottoposta a operazioni di post-processing morfologico.

La rimozione delle componenti connesse di area inferiore a `min_size = 20` può eliminare anche predizioni di piccola estensione qualora queste costituiscano componenti isolate.

Questo rappresenta un limite della procedura di post-processing e deve essere distinto dalla capacità del modello di generare la anomaly map continua.

### Rappresentazione delle maschere

Le maschere tumorali vengono utilizzate come maschere binarie per la valutazione pixel-level. La pipeline distingue quindi tra pixel appartenenti alla regione annotata e pixel appartenenti al background, senza preservare eventuali differenti sottoregioni presenti nell'annotazione originale.

### Generalizzazione

I risultati sono ottenuti sul dataset e sul protocollo sperimentale descritti nel progetto. Non è pertanto possibile assumere automaticamente che le prestazioni osservate si mantengano in presenza di dataset provenienti da centri differenti, scanner differenti o popolazioni cliniche differenti.

---

## 12. Sviluppi futuri

### Modelli 3D

Un'estensione naturale consiste nell'utilizzo di architetture 3D, come 3D-CNN e 3D Autoencoder, per sfruttare la continuità volumetrica tra slice adiacenti.

### Modelli generativi

È possibile valutare paradigmi ricostruttivi alternativi basati su modelli generativi più recenti, tra cui:

* Diffusion Models;
* Masked Autoencoders;
* architetture ibride generative e discriminative.

### Threshold sweep

Un'analisi sistematica delle soglie pixel-level potrebbe valutare l'andamento del Dice score e delle altre metriche di segmentazione al variare della soglia, mantenendo separata questa analisi dalla procedura operativa P99 utilizzata nel protocollo principale.

### Validazione esterna

Un'ulteriore direzione consiste nella validazione su coorti esterne, ad esempio altri dataset BraTS o dataset relativi ad altre patologie oncologiche cerebrali, al fine di valutare la capacità di generalizzazione dei modelli al di fuori della distribuzione utilizzata nello studio.

### Analisi per dimensione della lesione

Sarebbe inoltre possibile analizzare separatamente le prestazioni di localizzazione in funzione dell'area della lesione, per verificare quantitativamente l'impatto della dimensione della regione patologica sulle prestazioni pixel-level dei diversi modelli.
