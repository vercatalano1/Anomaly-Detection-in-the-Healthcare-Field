import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (roc_auc_score, roc_curve, average_precision_score, 
                             precision_recall_curve, confusion_matrix)
from dataloader import get_dataset

# Configurazione stile grafico per la tesi
sns.set_theme(style="whitegrid", font_scale=1.1)

def run_ml_baseline():
    print("\n" + "="*70)
    print(" ESPERIMENTO 1: Baseline Machine Learning - Analisi Esaustiva ")
    print("="*70)
    
    # Definizione della cartella di output specifica per questo modello
    output_dir = os.path.join("results", "isolationForest")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. CARICAMENTO DATI
    print("\n[1/4] Caricamento set di addestramento (solo cervelli sani)...")
    train_ds = get_dataset("brats", data_root="data", mode="train")
    
    print("\n[2/4] Caricamento set di test (sani e con tumore)...")
    test_ds = get_dataset("brats", data_root="data", mode="test")

    # 2. PREPARAZIONE DATI (FLATTENING)
    print("\n[3/4] Flattening dei tensori (da 64x64 a vettore 1D di 4096 feature)...")
    t0 = time.time()
    
    X_train = np.array([train_ds[i]['img'].numpy().flatten() for i in range(len(train_ds))])
    X_test = np.array([test_ds[i]['img'].numpy().flatten() for i in range(len(test_ds))])
    y_test = np.array([test_ds[i]['label'] for i in range(len(test_ds))])
    print(f"Flattening completato in {time.time() - t0:.2f} secondi.")

    # 3. ADDESTRAMENTO
    print("\n[4/4] Addestramento Isolation Forest in corso...")
    t_train = time.time()
    
    clf = IsolationForest(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train)
    time_addestramento = time.time() - t_train
    print(f"Addestramento ML completato in {time_addestramento:.2f} secondi!")

    # 4. INFERENZA E CALCOLO METRICHE AVANZATE
    print("\nEstrazione degli Anomaly Scores e calcolo metriche...")
    scores = -clf.decision_function(X_test)
    
    # Metriche Globali (Indipendenti dalla soglia)
    auroc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    
    # Ricerca della Soglia Ottimale (Best F1-Score)
    precisions, recalls, thresholds = precision_recall_curve(y_test, scores)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Binarizzazione delle predizioni usando la soglia ottimale
    y_pred = (scores >= best_threshold).astype(int)
    
    # Calcolo Matrice di Confusione e Metriche Cliniche
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Stampa a schermo
    print("\n" + "*" * 60)
    print(f" RISULTATI GLOBALI:")
    print(f" - AUROC                : {auroc:.4f}")
    print(f" - Average Precision (AP): {ap:.4f}")
    print("-" * 60)
    print(f" RISULTATI CLINICI (Soglia Ottimizzata = {best_threshold:.4f}):")
    print(f" - F1-Score             : {best_f1:.4f}")
    print(f" - Sensibilità (Recall) : {sensitivity:.4f} ({tp} trovati su {tp+fn})")
    print(f" - Specificità          : {specificity:.4f} ({tn} sani confermati su {tn+fp})")
    print("*" * 60)

    # 5. SALVATAGGIO AUTOMATICO DEL REPORT IN UN FILE DI TESTO
    report_path = os.path.join(output_dir, "report_metriche.txt")
    with open(report_path, "w") as f:
        f.write("==================================================\n")
        f.write("REPORT TESI: RISULTATI ISOLATION FOREST (BASELINE)\n")
        f.write("==================================================\n\n")
        f.write(f"Tempo di Addestramento : {time_addestramento:.4f} secondi\n\n")
        f.write("METRICHE GLOBALI:\n")
        f.write(f" - AUROC                : {auroc:.4f}\n")
        f.write(f" - Average Precision (AP): {ap:.4f}\n\n")
        f.write(f"METRICHE CLINICHE (Soglia Ottimale: {best_threshold:.4f}):\n")
        f.write(f" - F1-Score             : {best_f1:.4f}\n")
        f.write(f" - Sensibilità (Recall) : {sensitivity:.4f}\n")
        f.write(f" - Specificità          : {specificity:.4f}\n\n")
        f.write("MATRICE DI CONFUSIONE:\n")
        f.write(f" - Veri Negativi (Sani corretti)       : {tn}\n")
        f.write(f" - Falsi Positivi (Sani spaventati)    : {fp}\n")
        f.write(f" - Falsi Negativi (Tumori persi)       : {fn}\n")
        f.write(f" - Veri Positivi (Tumori scovati)      : {tp}\n")
    print(f"Report testuale salvato in: {report_path}")

    # 6. GENERAZIONE GRAFICI (Griglia 2x2 per la Tesi)
    print("\nGenerazione pannello grafico per il Capitolo 6...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # [A] Distribuzione degli Score
    sns.kdeplot(scores[y_test == 0], fill=True, color="green", label="Sani", ax=axes[0, 0], alpha=0.5)
    sns.kdeplot(scores[y_test == 1], fill=True, color="red", label="Tumori", ax=axes[0, 0], alpha=0.5)
    axes[0, 0].axvline(best_threshold, color='black', linestyle='--', label='Soglia Ottimale')
    axes[0, 0].set_title("A. Distribuzione Anomaly Score", fontweight='bold')
    axes[0, 0].set_xlabel("Anomaly Score")
    axes[0, 0].set_ylabel("Densità")
    axes[0, 0].legend()

    # [B] Matrice di Confusione
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], 
                xticklabels=['Sano', 'Tumore'], yticklabels=['Sano', 'Tumore'], annot_kws={"size": 14})
    axes[0, 1].set_title("B. Matrice di Confusione", fontweight='bold')
    axes[0, 1].set_xlabel("Predizione Modello")
    axes[0, 1].set_ylabel("Verità (Ground Truth)")

    # [C] Curva ROC
    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[1, 0].plot(fpr, tpr, color='blue', lw=2, label=f'AUROC = {auroc:.3f}')
    axes[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--')
    axes[1, 0].set_title("C. Curva ROC", fontweight='bold')
    axes[1, 0].set_xlabel("Tasso Falsi Positivi (1 - Specificità)")
    axes[1, 0].set_ylabel("Tasso Veri Positivi (Sensibilità)")
    axes[1, 0].legend(loc="lower right")

    # [D] Precision-Recall Curve
    random_baseline = sum(y_test) / len(y_test)
    axes[1, 1].plot(recalls, precisions, color='purple', lw=2, label=f'AP = {ap:.3f}')
    axes[1, 1].plot([0, 1], [random_baseline, random_baseline], color='gray', linestyle='--', label='Random')
    axes[1, 1].scatter(recalls[best_idx], precisions[best_idx], marker='o', color='red', s=100, label=f'Max F1 ({best_f1:.2f})')
    axes[1, 1].set_title("D. Precision-Recall Curve", fontweight='bold')
    axes[1, 1].set_xlabel("Recall (Sensibilità)")
    axes[1, 1].set_ylabel("Precisione")
    axes[1, 1].legend(loc="upper right")

    plt.tight_layout()
    grafico_path = os.path.join(output_dir, "ml_baseline_comprehensive.png")
    plt.savefig(grafico_path, dpi=300, bbox_inches='tight')
    print(f"Pannello grafico salvato con successo in: {grafico_path}")
    plt.show()

if __name__ == "__main__":
    run_ml_baseline()