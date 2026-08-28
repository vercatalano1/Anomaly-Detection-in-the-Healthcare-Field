# ============================================================
# COLLECT RESULTS — aggregazione automatica delle metriche
# ============================================================
#
# Legge i CSV di metriche gia' salvati da ciascun esperimento
# (ml_baseline, cnn_ae/*, CutPaste, PatchCore) e li normalizza in
# un unico schema comune, cosi' da eliminare la necessita' di
# copiare a mano i numeri in final_plot.py.
#
# Estrae inoltre il tempo di training/esecuzione dai report .txt
# (quando disponibile) tramite regex.
#
# Output:
#   results/summary/model_comparison.csv
#
# Esecuzione:
#   python src/analysis/collect_results.py
#
# Se una cartella/CSV non esiste ancora (esperimento non eseguito),
# la riga viene semplicemente saltata con un warning: lo script non
# si interrompe.
# ============================================================

import os
import re
# ============================================================
# COLLECT RESULTS — aggregazione automatica delle metriche
# ============================================================

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURAZIONE E PATH
# ============================================================
# Risaliamo l'albero partendo da dove si trova questo script
current_path = Path(__file__).resolve()

# Cerchiamo la cartella "src" per capire dov'è la radice
while current_path.name != "src" and current_path.parent != current_path:
    current_path = current_path.parent

# La radice del progetto è la cartella "padre" di "src"
if current_path.name == "src":
    PROJECT_ROOT = current_path.parent
else:
    # Fallback generico (adatta se lo script è a 2 livelli di profondità)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = RESULTS_DIR / "summary"

# ============================================================
# REGISTRO DELLE SORGENTI
# ============================================================
MODEL_SOURCES = [
    {
        "model": "Isolation Forest",
        "variant": "baseline",
        "metrics_csv": "ml_baseline/isolation_forest_metrics.csv",
        "report_txt": "ml_baseline/isolation_forest_report.txt",
        "schema": "iforest",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "baseline (MSE)",
        "metrics_csv": "cnn_autoencoder/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "ablation: loss=MSE",
        "metrics_csv": "cnn_autoencoder2_nofc_mse/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_mse/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "ablation: loss=L1",
        "metrics_csv": "cnn_autoencoder2_nofc_l1/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_l1/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "ablation: loss=MSE+L1",
        "metrics_csv": "cnn_autoencoder2_nofc_mse_l1/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_mse_l1/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "MSE+L1 + denoising gaussiano",
        "metrics_csv": "cnn_autoencoder2_nofc_mse_l1_denoising_gaussiano/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_mse_l1_denoising_gaussiano/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "MSE+L1 + coarse denoising 16x16",
        "metrics_csv": "cnn_autoencoder2_nofc_mse_l1_denoising_spaziale/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_mse_l1_denoising_spaziale/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CNN Autoencoder",
        "variant": "MSE+L1 + coarse denoising + post-processing",
        "metrics_csv": "cnn_autoencoder2_nofc_mse_l1_pp/cnn_autoencoder_metrics.csv",
        "report_txt": "cnn_autoencoder2_nofc_mse_l1_pp/cnn_autoencoder_report.txt",
        "schema": "cnn_ae",
    },
    {
        "model": "CutPaste",
        "variant": "self-supervised (image + pixel)",
        "metrics_csv": "cutpaste/cutpaste_metrics.csv",
        "report_txt": "cutpaste/cutpaste_report.txt",
        "schema": "cutpaste_image",
    },
    {
        "model": "CutPaste",
        "variant": "self-supervised (image + pixel)",
        "metrics_csv": "cutpaste/pixel_level/pixel_level_metrics.csv",
        "report_txt": "cutpaste/pixel_level/pixel_level_report.txt",
        "schema": "cutpaste_pixel",
        "merge": True,
    },
    {
        "model": "PatchCore",
        "variant": "final",
        "metrics_csv": "patchcore/patchcore_image_metrics.csv",
        "report_txt": "patchcore/patchcore_report.txt",
        "schema": "patchcore_image",
    },
    {
        "model": "PatchCore",
        "variant": "final",
        "metrics_csv": "patchcore/patchcore_pixel_metrics.csv",
        "report_txt": "patchcore/patchcore_report.txt",
        "schema": "patchcore_pixel",
        "merge": True,
    },
]

SCHEMA_MAPS: Dict[str, Dict[str, str]] = {
    "iforest": {
        "image_auroc": "AUROC",
        "image_ap": "Average_Precision",
        "image_f1": "F1",
        "image_sensitivity": "Sensitivity",
        "image_specificity": "Specificity",
        "image_bacc": "Balanced_Accuracy",
    },
    "cnn_ae": {
        "image_auroc": "Image_AUROC",
        "image_ap": "Image_Average_Precision",
        "image_f1": "Image_F1",
        "image_sensitivity": "Image_Sensitivity",
        "image_specificity": "Image_Specificity",
        "image_bacc": "Image_Balanced_Accuracy",
        "pixel_auroc": "Pixel_AUROC",
        "pixel_ap": "Pixel_Average_Precision",
        "pixel_dice": "Pixel_Dice",
        "pixel_iou": "Pixel_IoU",
        "pixel_sensitivity": "Pixel_Sensitivity",
        "pixel_specificity": "Pixel_Specificity",
    },
    "cutpaste_image": {
        "image_auroc": "AUROC",
        "image_ap": "Average_Precision",
        "image_f1": "F1",
        "image_sensitivity": "Sensitivity",
        "image_specificity": "Specificity",
        "image_bacc": "Balanced_Accuracy",
    },
    "cutpaste_pixel": {
        "pixel_auroc": "pixel_auroc",
        "pixel_ap": "pixel_ap",
        "pixel_dice": "pixel_dice",
        "pixel_iou": "pixel_iou",
        "pixel_sensitivity": "pixel_sensitivity",
        "pixel_specificity": "pixel_specificity",
    },
    "patchcore_image": {
        "image_auroc": "auroc",
        "image_ap": "ap",
        "image_f1": "f1",
        "image_sensitivity": "sensitivity",
        "image_specificity": "specificity",
        "image_bacc": "bacc",
    },
    "patchcore_pixel": {
        "pixel_auroc": "pixel_auroc",
        "pixel_ap": "pixel_ap",
        "pixel_dice": "pixel_dice",
        "pixel_iou": "pixel_iou",
        "pixel_sensitivity": "pixel_sensitivity",
        "pixel_specificity": "pixel_specificity",
    },
}

TIME_PATTERNS = [
    r"Training time:\s*([\d.]+)s",
    r"Execution time:\s*([\d.]+)s",
    r"Computation time:\s*([\d.]+)s",
]

def extract_time_seconds(report_path: Optional[str]) -> Optional[float]:
    if report_path is None:
        return None
    full_path = os.path.join(RESULTS_DIR, report_path)
    if not os.path.isfile(full_path):
        return None
    with open(full_path, "r", encoding="utf-8") as f:
        text = f.read()
    for pattern in TIME_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None

def load_and_normalize(source: Dict) -> Optional[Dict]:
    csv_path = os.path.join(RESULTS_DIR, source["metrics_csv"])
    if not os.path.isfile(csv_path):
        print(f"  [SKIP] {source['model']} / {source['variant']} — file non trovato: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    mapping = SCHEMA_MAPS[source["schema"]]
    normalized = {}
    for target_col, source_col in mapping.items():
        if source_col in row.index:
            normalized[target_col] = row[source_col]
        else:
            normalized[target_col] = float("nan")
    normalized["training_time_s"] = extract_time_seconds(source.get("report_txt"))
    normalized["_model"] = source["model"]
    normalized["_variant"] = source["variant"]
    normalized["_merge"] = source.get("merge", False)
    return normalized

def collect() -> pd.DataFrame:
    print("=" * 70)
    print("COLLECT RESULTS — aggregazione metriche")
    print("=" * 70)
    print(f"\nCartella risultati: {RESULTS_DIR}\n")

    rows_by_key = {}
    for source in MODEL_SOURCES:
        normalized = load_and_normalize(source)
        if normalized is None:
            continue
        key = (normalized["_model"], normalized["_variant"])
        if key not in rows_by_key:
            rows_by_key[key] = {
                "model": normalized["_model"],
                "variant": normalized["_variant"],
            }
        target = rows_by_key[key]

        for col, value in normalized.items():

            if col.startswith("_"):
                continue

            # Somma i tempi se ci sono più report per lo stesso modello/variante (es. CutPaste Image + Pixel)
            if col == "training_time_s":
                current_time = target.get(col, 0.0)
                if current_time is None:
                    current_time = 0.0
                if value is not None:
                    target[col] = current_time + value
                continue

            if pd.notnull(value):
                target[col] = value
        print(f"  [OK]   {normalized['_model']} / {normalized['_variant']} ({source['schema']})")

    if not rows_by_key:
        print("\nNessun risultato trovato.")
        sys.exit(1)

    summary_df = pd.DataFrame(list(rows_by_key.values()))
    ordered_cols = [
        "model", "variant",
        "image_auroc", "image_ap", "image_f1",
        "image_sensitivity", "image_specificity", "image_bacc",
        "pixel_auroc", "pixel_ap", "pixel_dice", "pixel_iou",
        "pixel_sensitivity", "pixel_specificity",
        "training_time_s",
    ]
    ordered_cols = [c for c in ordered_cols if c in summary_df.columns]
    summary_df = summary_df[ordered_cols]
    summary_df = summary_df.reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "model_comparison.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n✓ Salvato CSV: {out_path}")

    # ============================================================
    # SALVATAGGIO TABELLA GRAFICA (PNG) — Con valori migliori in evidenza
    # ============================================================
    
    rename_columns = {
        "model": "Model",
        "variant": "Variant",
        "image_auroc": "Img AUROC",
        "image_ap": "Img AP",
        "image_f1": "Img F1",
        "image_sensitivity": "Img Sens",
        "image_specificity": "Img Spec",
        "image_bacc": "Img BAcc",
        "pixel_auroc": "Pix AUROC",
        "pixel_ap": "Pix AP",
        "pixel_dice": "Pixel Dice",
        "pixel_iou": "Pixel IoU",
        "pixel_sensitivity": "Pix Sens",
        "pixel_specificity": "Pix Spec",
        "training_time_s": "Time (s)"
    }
    
    plot_df = summary_df.rename(columns=rename_columns).copy()
    
    # Individuiamo i valori massimi e minimi per le colonne numeriche prima di trasformarle in stringhe
    numeric_cols = [c for c in plot_df.columns if c not in ["Model", "Variant"]]
    
    best_values = {}
    for col in numeric_cols:
        temp_col = pd.to_numeric(plot_df[col], errors='coerce')
        if not temp_col.dropna().empty:
            if col == "Time (s)":
                best_values[col] = temp_col.min() # Il più basso per il tempo
            else:
                best_values[col] = temp_col.max() # Il più alto per le metriche

    # Ora convertiamo tutto il DataFrame in stringa per evitare conflitti di tipo con Pandas
    formatted_df = plot_df.astype(str)
    
    for col in numeric_cols:
        temp_col = pd.to_numeric(plot_df[col], errors='coerce')
        if col in best_values:
            best_val = best_values[col]
            for idx in plot_df.index:
                val = temp_col.loc[idx]
                if pd.notnull(val):
                    if np.isclose(val, best_val, atol=1e-4):
                        formatted_df.loc[idx, col] = f"mathbf{{{val:.4f}}}"
                    else:
                        formatted_df.loc[idx, col] = f"{val:.4f}"
                else:
                    formatted_df.loc[idx, col] = "—"

    # Allarghiamo la figura
    fig, ax = plt.subplots(figsize=(22, len(summary_df) * 0.7 + 3))
    ax.axis("off")
    ax.axis("tight")

    col_widths = [0.12, 0.22] + [0.046] * (len(plot_df.columns) - 2)

    table = ax.table(
        cellText=formatted_df.values,
        colLabels=plot_df.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.8)

    # Stile dell'intestazione
    for j, col_name in enumerate(plot_df.columns):
        cell = table[(0, j)]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(weight="bold", color="white")

    # Scansione delle celle per applicare il grassetto e lo sfondo verde ai record migliori
    for i in range(1, len(formatted_df) + 1):
        for j in range(len(plot_df.columns)):
            cell = table[(i, j)]
            cell_text = cell.get_text().get_text()
            
            if "mathbf" in cell_text:
                clean_text = cell_text.replace("mathbf{", "").replace("}", "")
                cell.get_text().set_text(clean_text)
                cell.get_text().set_weight("bold")
                cell.set_facecolor("#d4edda") # Sfondo verde chiaro per i top score
            else:
                if i % 2 == 0:
                    cell.set_facecolor("#f8f9fa")
                else:
                    cell.set_facecolor("#ffffff")

    plt.title("Comparative Summary — All Models & Variants (Best in Bold)", fontweight="bold", fontsize=16, pad=25)
    
    img_out_path = os.path.join(OUT_DIR, "model_comparison.png")
    plt.savefig(img_out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"✓ Salvata Tabella con evidenziazione dei valori migliori: {img_out_path}\n")
    print(summary_df.to_string(index=False))

    return summary_df

if __name__ == "__main__":
    collect()