# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS & PAIRWISE AUROC TEST
# (SLICE-LEVEL — analisi statistica primaria)
# ============================================================
#
# Analisi statistica finale e complementare:
#   1. Bootstrap IC95% per AUROC e Average Precision
#   2. Test di DeLong per confronti appaiati delle AUROC
#
# UNITÀ STATISTICA: slice (osservazione = singola immagine).
#
# NOTA METODOLOGICA:
#   Questa è l'unità statistica scelta come riferimento primario
#   per l'inferenza (Sezione "Scelta del livello di aggregazione
#   per l'inferenza statistica" nel capitolo Valutazione), poiché
#   a livello di paziente il test set presenta una numerosità
#   della classe minoritaria estremamente ridotta (un solo
#   paziente privo di slice tumorali, si veda la Sezione EDA sul
#   controllo del data leakage), che rende il bootstrap
#   patient-level poco informativo (script separato,
#   statistical_patient_level.py, riportato come analisi
#   complementare).
#
#   Per verificare quantitativamente che questo problema NON
#   riguardi il livello di slice, ogni bootstrap qui registra
#   esplicitamente il numero di iterazioni valide (cioè con
#   entrambe le classi rappresentate) su quelle richieste.
#   Con 828 slice normal e 1.948 slice tumor, ci si attende che
#   la quasi totalità delle iterazioni risulti valida.
#
# Confronti:
#   - PatchCore vs CNN Autoencoder
#   - PatchCore vs Isolation Forest
#
# ============================================================

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIGURAZIONE E PATH
# ============================================================

current_path = Path(__file__).resolve()

while current_path.name != "src" and current_path.parent != current_path:
    current_path = current_path.parent

if current_path.name == "src":
    PROJECT_ROOT = current_path.parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "summary" / "bootstrap_analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURAZIONE STATISTICA
# ============================================================

N_BOOTSTRAP = 2000
SEED = 42
ALPHA = 0.05  # IC 95%

# Soglia sotto la quale il numero di bootstrap validi viene
# segnalato esplicitamente come potenzialmente problematico.
MIN_VALID_FRACTION_WARNING = 0.8

REFERENCE_MODEL_NAME = "PatchCore"

SOURCES = [
    (
        "Isolation Forest",
        "ml_baseline/error_analysis.csv"
    ),
    (
        "CNN Autoencoder",
        "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"
    ),
    (
        "PatchCore",
        "patchcore/image_level_results.csv"
    ),
]

ORDER_MODELS = [
    "PatchCore",
    "CNN Autoencoder",
    "Isolation Forest"
]

MODEL_COLORS = {
    "Isolation Forest": "#d62728",  # Rosso morbido
    "CNN Autoencoder": "#2ca02c",   # Verde
    "PatchCore": "#1f77b4"          # Blu
}


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_metric(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED
) -> Tuple[float, float, float, int, int]:
    """
    Restituisce (point_estimate, lower, upper, n_valid, n_requested).

    n_valid conta le iterazioni bootstrap in cui il campione
    ricampionato contiene ENTRAMBE le classi (unica condizione
    sotto cui AUROC/AP sono definite). A livello di slice, con
    828 negativi e 1.948 positivi, n_valid è atteso prossimo a
    n_requested; un valore sensibilmente inferiore segnalerebbe
    un problema di numerosità che qui non ci si aspetta di
    osservare (a differenza del livello di paziente).
    """

    rng = np.random.default_rng(seed)

    n = len(y_true)

    point_estimate = metric_fn(y_true, scores)

    bootstrap_values = []

    for _ in range(n_bootstrap):

        indices = rng.integers(0, n, size=n)

        y_boot = y_true[indices]
        scores_boot = scores[indices]

        # AUROC/AP non sono definiti se rimane una sola classe
        if len(np.unique(y_boot)) < 2:
            continue

        bootstrap_values.append(
            metric_fn(y_boot, scores_boot)
        )

    bootstrap_values = np.asarray(bootstrap_values)

    n_valid = len(bootstrap_values)

    if n_valid == 0:
        raise RuntimeError(
            "Nessun bootstrap valido disponibile."
        )

    lower = np.percentile(
        bootstrap_values,
        100 * ALPHA / 2
    )

    upper = np.percentile(
        bootstrap_values,
        100 * (1 - ALPHA / 2)
    )

    return point_estimate, lower, upper, n_valid, n_bootstrap


# ============================================================
# DELONG TEST
# ============================================================

def compute_midrank(x):
    """
    Calcola i midrank utilizzati dal test di DeLong.
    """
    x = np.asarray(x)

    order = np.argsort(x)
    sorted_x = x[order]

    midranks = np.empty(len(x), dtype=float)

    i = 0

    while i < len(sorted_x):

        j = i

        while j < len(sorted_x) and sorted_x[j] == sorted_x[i]:
            j += 1

        midranks[order[i:j]] = 0.5 * (i + j - 1) + 1

        i = j

    return midranks


def fast_delong(
    predictions_sorted_transposed: np.ndarray,
    label_1_count: int
):
    """
    Implementazione del calcolo della varianza DeLong
    per ROC curve correlate.
    """

    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m

    positive_predictions = predictions_sorted_transposed[:, :m]
    negative_predictions = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))

    tz = np.empty((k, m + n))

    for r in range(k):
        tx[r] = compute_midrank(positive_predictions[r])
        ty[r] = compute_midrank(negative_predictions[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)

    if k == 1:
        sx = np.asarray([[sx]])
        sy = np.asarray([[sy]])

    covariance = sx / m + sy / n

    return aucs, covariance


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Tuple[float, float, float]:
    """
    Test di DeLong per confrontare due AUROC correlate.

    Restituisce: auc_a, auc_b, p_value
    """

    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    if len(y_true) != len(scores_a) or len(y_true) != len(scores_b):
        raise ValueError(
            "y_true e gli score devono avere la stessa lunghezza."
        )

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "Il test di DeLong richiede entrambe le classi."
        )

    order = np.argsort(-y_true)

    y_sorted = y_true[order]

    predictions = np.vstack([
        scores_a[order],
        scores_b[order]
    ])

    positive_count = int(np.sum(y_sorted))

    aucs, covariance = fast_delong(predictions, positive_count)

    auc_a = aucs[0]
    auc_b = aucs[1]

    variance_difference = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )

    if variance_difference <= 0:
        p_value = 1.0

    else:
        z = abs(auc_a - auc_b) / np.sqrt(variance_difference)

        from math import erf, sqrt

        p_value = 1.0 - erf(z / sqrt(2.0))

    return auc_a, auc_b, p_value


# ============================================================
# PAIRED BOOTSTRAP PER DIFFERENZA AUROC
# ============================================================

def paired_bootstrap_delta(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_challenger: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED
) -> Tuple[float, float, float, int, int]:
    """
    Restituisce (delta_point, lower, upper, n_valid, n_requested).
    """

    rng = np.random.default_rng(seed)

    n = len(y_true)

    delta_point = (
        roc_auc_score(y_true, scores_ref)
        -
        roc_auc_score(y_true, scores_challenger)
    )

    deltas = []

    for _ in range(n_bootstrap):

        indices = rng.integers(0, n, size=n)

        y_boot = y_true[indices]

        if len(np.unique(y_boot)) < 2:
            continue

        auc_ref = roc_auc_score(y_boot, scores_ref[indices])
        auc_challenger = roc_auc_score(y_boot, scores_challenger[indices])

        deltas.append(auc_ref - auc_challenger)

    deltas = np.asarray(deltas)

    n_valid = len(deltas)

    if n_valid == 0:
        raise RuntimeError(
            "Nessun bootstrap valido per la differenza AUROC."
        )

    lower = np.percentile(deltas, 100 * ALPHA / 2)
    upper = np.percentile(deltas, 100 * (1 - ALPHA / 2))

    return delta_point, lower, upper, n_valid, n_bootstrap


# ============================================================
# FOREST PLOT
# ============================================================

def plot_forest(
    df: pd.DataFrame,
    metric: str,
    title: str,
    filename: str
):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    df_plot = (
        df
        .set_index("model")
        .reindex(ORDER_MODELS)
        .reset_index()
    )

    y_pos = np.arange(len(df_plot))

    # Sfondo a strisce morbide per facilitare la lettura
    for i in range(len(df_plot)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f8f9fa', zorder=0)

    points = df_plot[metric].to_numpy()
    lower_errors = points - df_plot[f"{metric}_ci_low"].to_numpy()
    upper_errors = df_plot[f"{metric}_ci_high"].to_numpy() - points

    # Linea verticale di riferimento sul modello di riferimento (PatchCore)
    if REFERENCE_MODEL_NAME in df_plot["model"].values:
        best_val = df_plot.loc[df_plot["model"] == REFERENCE_MODEL_NAME, metric].values[0]
        ax.axvline(best_val, linestyle=":", color=MODEL_COLORS.get(REFERENCE_MODEL_NAME, "gray"), linewidth=2, zorder=1, alpha=0.6)

    # Disegna ogni punto con il suo colore specifico
    for i, model in enumerate(df_plot["model"]):
        color = MODEL_COLORS.get(model, "#333333")
        
        ax.errorbar(
            points[i], i, 
            xerr=[[lower_errors[i]], [upper_errors[i]]],
            fmt="D", markersize=9, capsize=6, capthick=2.5, elinewidth=2.5, 
            color=color, zorder=3, markeredgecolor='white', markeredgewidth=1
        )

        value = points[i]
        low = df_plot.loc[i, f"{metric}_ci_low"]
        high = df_plot.loc[i, f"{metric}_ci_high"]
        n_valid = df_plot.loc[i, f"{metric}_n_valid_bootstrap"]
        n_req = df_plot.loc[i, f"{metric}_n_requested_bootstrap"]

        # Testo del CI a destra
        ax.text(
            max(points) + max(upper_errors) + 0.02, i, 
            f"{value:.3f} [{low:.3f}, {high:.3f}]  (n={n_valid}/{n_req})", 
            va="center", fontsize=9, color='#333333'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["model"], fontweight="bold", fontsize=11)
    
    # Colora le etichette dell'asse Y per richiamare i modelli
    for label in ax.get_yticklabels():
        label.set_color(MODEL_COLORS.get(label.get_text(), "#333333"))

    ax.set_xlabel(f"{metric.upper()} con 95% CI", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", pad=15, fontsize=13)
    
    min_ci = df_plot[f"{metric}_ci_low"].min()
    max_ci = df_plot[f"{metric}_ci_high"].max()
    ax.set_xlim(max(0, min_ci - 0.05), min(1.0, max_ci + 0.20))

    ax.grid(axis="x", linestyle="--", alpha=0.5, color='gray', zorder=1)
    ax.grid(axis="y", visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# PAIRWISE FOREST PLOT
# ============================================================

def plot_pairwise_forest(df_pair: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    y_pos = np.arange(len(df_pair))
    
    points = df_pair["Delta_AUROC"].to_numpy()
    lower_errors = points - df_pair["CI_Lower"].to_numpy()
    upper_errors = df_pair["CI_Upper"].to_numpy() - points

    # Sfondo a strisce morbide
    for i in range(len(df_pair)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f8f9fa', zorder=0)

    # Linea dello zero di riferimento
    ax.axvline(0, linestyle="--", color="#333333", linewidth=1.5, alpha=0.8, zorder=1)

    for i, model in enumerate(df_pair["Challenger"]):
        color = MODEL_COLORS.get(model, "#333333")
        
        ax.errorbar(
            points[i], i, 
            xerr=[[lower_errors[i]], [upper_errors[i]]],
            fmt="s", markersize=9, capsize=6, capthick=2.5, elinewidth=2.5, 
            color=color, zorder=3, markeredgecolor='white', markeredgewidth=1
        )

        delta = points[i]
        low = df_pair.loc[i, "CI_Lower"]
        high = df_pair.loc[i, "CI_Upper"]
        p_value = df_pair.loc[i, "p_value"]
        n_valid = df_pair.loc[i, "n_valid_bootstrap"]
        n_req = df_pair.loc[i, "n_requested_bootstrap"]

        # Formattazione per la significatività
        if p_value < 0.001:
            sig_stars = "***"
            p_text = "< 0.001"
        elif p_value < 0.01:
            sig_stars = "**"
            p_text = f"= {p_value:.3f}"
        elif p_value < 0.05:
            sig_stars = "*"
            p_text = f"= {p_value:.3f}"
        else:
            sig_stars = "(ns)"
            p_text = f"= {p_value:.3f}"

        ax.text(
            max(points) + max(upper_errors) + 0.02, i, 
            f"Δ={delta:.3f} [{low:.3f}, {high:.3f}] | p {p_text} {sig_stars}  (n={n_valid}/{n_req})", 
            va="center", fontsize=9, color='#333333'
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pair["Challenger"], fontweight="bold", fontsize=11)
    
    # Colora le etichette dell'asse Y
    for label in ax.get_yticklabels():
        label.set_color(MODEL_COLORS.get(label.get_text(), "#333333"))

    ax.set_xlabel(f"Δ AUROC ({REFERENCE_MODEL_NAME} − Modello Challenger)", fontweight="bold", fontsize=11)
    ax.set_title(f"Test di DeLong: {REFERENCE_MODEL_NAME} vs Altri Modelli", fontweight="bold", pad=15, fontsize=13)
    
    ax.grid(axis="x", linestyle="--", alpha=0.5, color='gray', zorder=1)
    ax.grid(axis="y", visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "forest_plot_pairwise.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  ✓ Salvato: forest_plot_pairwise.png")


# ============================================================
# ESECUZIONE PRINCIPALE
# ============================================================

def run():

    print("=" * 70)
    print("BOOTSTRAP CONFIDENCE INTERVALS & DELONG TEST (SLICE-LEVEL)")
    print(f"{N_BOOTSTRAP} resampling | IC {100 * (1 - ALPHA):.0f}%")
    print("=" * 70)

    rows = []
    model_data = {}

    # --------------------------------------------------------
    # 1. CARICAMENTO DATI E BOOTSTRAP
    # --------------------------------------------------------

    for model_name, relative_path in SOURCES:

        csv_path = RESULTS_DIR / relative_path

        if not csv_path.exists():
            print(f"  [SKIP] {model_name}: {csv_path} non trovato")
            continue

        df = pd.read_csv(csv_path)

        required_columns = {"true_label", "anomaly_score"}
        missing = required_columns - set(df.columns)

        if missing:
            print(f"  [SKIP] {model_name}: colonne mancanti {missing}")
            continue

        y_true = df["true_label"].to_numpy()
        scores = df["anomaly_score"].to_numpy()

        if len(np.unique(y_true)) < 2:
            print(f"  [SKIP] {model_name}: presente una sola classe")
            continue

        n_negative = int(np.sum(y_true == 0))
        n_positive = int(np.sum(y_true == 1))

        model_data[model_name] = df

        # AUROC
        auroc, auroc_lo, auroc_hi, auroc_n_valid, auroc_n_req = (
            bootstrap_metric(y_true, scores, roc_auc_score)
        )

        # Average Precision
        ap, ap_lo, ap_hi, ap_n_valid, ap_n_req = (
            bootstrap_metric(y_true, scores, average_precision_score)
        )

        rows.append({
            "model": model_name,
            "n_slices": len(df),
            "n_negative": n_negative,
            "n_positive": n_positive,

            "auroc": auroc,
            "auroc_ci_low": auroc_lo,
            "auroc_ci_high": auroc_hi,
            "auroc_n_valid_bootstrap": auroc_n_valid,
            "auroc_n_requested_bootstrap": auroc_n_req,

            "ap": ap,
            "ap_ci_low": ap_lo,
            "ap_ci_high": ap_hi,
            "ap_n_valid_bootstrap": ap_n_valid,
            "ap_n_requested_bootstrap": ap_n_req,
        })

        print(
            f"  [OK] {model_name:18s} | "
            f"n = {len(df)} (neg={n_negative}, pos={n_positive}) | "
            f"AUROC = {auroc:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}] "
            f"(valid={auroc_n_valid}/{auroc_n_req}) | "
            f"AP = {ap:.4f} [{ap_lo:.4f}, {ap_hi:.4f}] "
            f"(valid={ap_n_valid}/{ap_n_req})"
        )

        if auroc_n_valid < MIN_VALID_FRACTION_WARNING * auroc_n_req:
            print(
                f"  [WARNING] {model_name}: solo {auroc_n_valid}/{auroc_n_req} "
                f"bootstrap validi per l'AUROC — verificare la numerosità "
                f"delle classi in questo campione."
            )

    if not rows:
        print("\nNessun risultato trovato.")
        return

    result_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # SALVATAGGIO IC
    # --------------------------------------------------------

    result_df.to_csv(
        OUT_DIR / "bootstrap_confidence_intervals.csv",
        index=False
    )

    print("\n✓ Salvato: bootstrap_confidence_intervals.csv")

    # --------------------------------------------------------
    # FOREST PLOTS
    # --------------------------------------------------------

    print("\nGenerazione Forest Plots...")

    plot_forest(
        result_df, "auroc",
        "AUROC con Intervalli di Confidenza al 95% (slice-level)",
        "forest_plot_auroc.png"
    )

    plot_forest(
        result_df, "ap",
        "Average Precision con Intervalli di Confidenza al 95% (slice-level)",
        "forest_plot_ap.png"
    )

    print("  ✓ Salvato: forest_plot_auroc.png")
    print("  ✓ Salvato: forest_plot_ap.png")

    # --------------------------------------------------------
    # 2. CONFRONTI PAIRWISE
    # --------------------------------------------------------

    if REFERENCE_MODEL_NAME not in model_data:
        print(f"\n[WARNING] {REFERENCE_MODEL_NAME} non disponibile.")
        return

    print("\n" + "=" * 70)
    print(f"PAIRWISE AUROC TEST (slice-level): {REFERENCE_MODEL_NAME} vs Altri")
    print("=" * 70)

    df_ref = model_data[REFERENCE_MODEL_NAME]

    y_ref = df_ref["true_label"].to_numpy()
    scores_ref = df_ref["anomaly_score"].to_numpy()

    pairwise_results = []

    for model_name, df_chal in model_data.items():

        if model_name == REFERENCE_MODEL_NAME:
            continue

        y_chal = df_chal["true_label"].to_numpy()
        scores_chal = df_chal["anomaly_score"].to_numpy()

        # ----------------------------------------------------
        # CONTROLLO: stesso numero di campioni e stesse label
        # (allineamento posizionale valido poiché tutte le
        # pipeline iterano sullo stesso test_ds ordinato,
        # shuffle=False)
        # ----------------------------------------------------

        if len(y_ref) != len(y_chal):
            print(f"  [SKIP] vs {model_name}: numero di campioni differente")
            continue

        if not np.array_equal(y_ref, y_chal):
            print(f"  [SKIP] vs {model_name}: le label non sono allineate")
            continue

        # ----------------------------------------------------
        # DeLong
        # ----------------------------------------------------

        auc_ref, auc_chal, p_value = delong_test(y_ref, scores_ref, scores_chal)

        # ----------------------------------------------------
        # Bootstrap paired della differenza
        # ----------------------------------------------------

        delta, ci_low, ci_high, delta_n_valid, delta_n_req = (
            paired_bootstrap_delta(
                y_ref, scores_ref, scores_chal,
                n_bootstrap=N_BOOTSTRAP, seed=SEED
            )
        )

        significant = "SI" if p_value < ALPHA else "NO"

        pairwise_results.append({
            "Reference": REFERENCE_MODEL_NAME,
            "Challenger": model_name,
            "AUROC_Reference": auc_ref,
            "AUROC_Challenger": auc_chal,
            "Delta_AUROC": delta,
            "CI_Lower": ci_low,
            "CI_Upper": ci_high,
            "n_valid_bootstrap": delta_n_valid,
            "n_requested_bootstrap": delta_n_req,
            "p_value": p_value,
            "Significant": significant,
        })

        print(
            f"  [OK] vs {model_name:18s} | "
            f"Δ AUROC = {delta:.4f} | "
            f"95% CI = [{ci_low:.4f}, {ci_high:.4f}] "
            f"(valid={delta_n_valid}/{delta_n_req}) | "
            f"p = {p_value:.4f} | Sig = {significant}"
        )

        if delta_n_valid < MIN_VALID_FRACTION_WARNING * delta_n_req:
            print(
                f"  [WARNING] vs {model_name}: solo "
                f"{delta_n_valid}/{delta_n_req} bootstrap validi per Δ AUROC."
            )

    # --------------------------------------------------------
    # SALVATAGGIO RISULTATI PAIRWISE
    # --------------------------------------------------------

    if pairwise_results:

        df_pair = pd.DataFrame(pairwise_results)

        df_pair.to_csv(
            OUT_DIR / "pairwise_statistical_test.csv",
            index=False
        )

        print("\n✓ Salvato: pairwise_statistical_test.csv")

        plot_pairwise_forest(df_pair)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run()





# ============================================================
# PATIENT-LEVEL STATISTICAL ANALYSIS
# ============================================================
#
# Analisi statistica complementare alla valutazione patient-level:
#
#   1. Bootstrap IC95% per AUROC e Average Precision
#   2. Test di DeLong per confronti appaiati delle AUROC
#   3. Paired bootstrap per la differenza di AUROC
#
# UNITÀ STATISTICA:
#   Paziente
#
# REGOLA DI AGGREGAZIONE:
#   patient_score = max(anomaly_score) sulle slice del paziente
#   patient_label = max(true_label)
#
# CONFRONTI:
#   - PatchCore vs CNN Autoencoder
#   - PatchCore vs Isolation Forest
#
# NOTA:
#   AUROC e AP patient-level NON vengono ricalcolate come
#   analisi principale: vengono ricostruiti gli stessi
#   patient_score/patient_label dello script patient_level.py
#   esclusivamente per effettuare l'inferenza statistica.
#
# ============================================================


'''import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


# ============================================================
# CONFIGURAZIONE
# ============================================================

N_BOOTSTRAP = 2000
SEED = 42
ALPHA = 0.05

REFERENCE_MODEL_NAME = "PatchCore"

MODEL_ORDER = [
    "PatchCore",
    "CNN Autoencoder",
    "Isolation Forest"
]


SOURCES = [
    (
        "Isolation Forest",
        "ml_baseline/error_analysis.csv"
    ),
    (
        "CNN Autoencoder",
        "cnn_autoencoder2_nofc_mse_l1_pp/image_level_results.csv"
    ),
    (
        "PatchCore",
        "patchcore/image_level_results.csv"
    ),
]


MODEL_COLORS = {
    "Isolation Forest": "#d62728",
    "CNN Autoencoder": "#2ca02c",
    "PatchCore": "#1f77b4"
}


# ============================================================
# PROJECT PATHS
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent


def find_project_root(current: Path) -> Path:

    for parent in [current] + list(current.parents):

        if (parent / "results").exists():
            return parent

    return current.parent


PROJECT_ROOT = find_project_root(CURRENT_DIR)

RESULTS_DIR = PROJECT_ROOT / "results"

OUT_DIR = (
    RESULTS_DIR
    / "summary"
    / "patient_level"
    / "statistical_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


plt.rcParams.update({
    "font.size": 11
})


# ============================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================

def compute_patient_level(
    df: pd.DataFrame
) -> pd.DataFrame:

    required = {
        "patient_id",
        "true_label",
        "anomaly_score"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Colonne mancanti nel CSV: {missing}"
        )

    patient_df = (
        df
        .groupby("patient_id")
        .agg(
            patient_label=(
                "true_label",
                "max"
            ),
            patient_score=(
                "anomaly_score",
                "max"
            ),
            n_slices=(
                "true_label",
                "count"
            )
        )
        .reset_index()
    )

    return patient_df


# ============================================================
# BOOTSTRAP GENERICO
# ============================================================

def bootstrap_metric(
    y_true,
    scores,
    metric_fn,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED
):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    rng = np.random.default_rng(seed)

    n = len(y_true)

    point_estimate = metric_fn(
        y_true,
        scores
    )

    bootstrap_values = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_boot = y_true[indices]
        scores_boot = scores[indices]

        # Evita campioni contenenti una sola classe
        if len(np.unique(y_boot)) < 2:
            continue

        value = metric_fn(
            y_boot,
            scores_boot
        )

        bootstrap_values.append(value)

    bootstrap_values = np.asarray(
        bootstrap_values
    )

    if len(bootstrap_values) == 0:

        raise RuntimeError(
            "Nessun bootstrap valido disponibile."
        )

    lower = np.percentile(
        bootstrap_values,
        100 * ALPHA / 2
    )

    upper = np.percentile(
        bootstrap_values,
        100 * (1 - ALPHA / 2)
    )

    return (
        point_estimate,
        lower,
        upper
    )


# ============================================================
# DELONG TEST
# ============================================================

def compute_midrank(x):

    x = np.asarray(x)

    order = np.argsort(x)

    sorted_x = x[order]

    midranks = np.empty(
        len(x),
        dtype=float
    )

    i = 0

    while i < len(sorted_x):

        j = i

        while (
            j < len(sorted_x)
            and sorted_x[j] == sorted_x[i]
        ):
            j += 1

        midranks[
            order[i:j]
        ] = (
            0.5 * (i + j - 1)
            + 1
        )

        i = j

    return midranks


def fast_delong(
    predictions_sorted_transposed,
    label_1_count
):

    m = label_1_count

    n = (
        predictions_sorted_transposed.shape[1]
        - m
    )

    positive_predictions = (
        predictions_sorted_transposed[:, :m]
    )

    negative_predictions = (
        predictions_sorted_transposed[:, m:]
    )

    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))

    for r in range(k):

        tx[r] = compute_midrank(
            positive_predictions[r]
        )

        ty[r] = compute_midrank(
            negative_predictions[r]
        )

        tz[r] = compute_midrank(
            predictions_sorted_transposed[r]
        )

    aucs = (
        tz[:, :m].sum(axis=1)
        / (m * n)
        - (m + 1.0)
        / (2.0 * n)
    )

    v01 = (
        tz[:, :m] - tx
    ) / n

    v10 = (
        1.0
        - (
            tz[:, m:] - ty
        ) / m
    )

    sx = np.cov(v01)

    sy = np.cov(v10)

    if k == 1:

        sx = np.asarray([[sx]])
        sy = np.asarray([[sy]])

    covariance = (
        sx / m
        + sy / n
    )

    return aucs, covariance


def delong_test(
    y_true,
    scores_a,
    scores_b
):

    y_true = np.asarray(y_true)

    scores_a = np.asarray(scores_a)

    scores_b = np.asarray(scores_b)

    order = np.argsort(
        -y_true
    )

    y_sorted = y_true[order]

    predictions = np.vstack([
        scores_a[order],
        scores_b[order]
    ])

    positive_count = int(
        np.sum(y_sorted)
    )

    if positive_count == 0:

        raise ValueError(
            "Nessuna classe positiva."
        )

    if positive_count == len(y_sorted):

        raise ValueError(
            "Nessuna classe negativa."
        )

    aucs, covariance = fast_delong(
        predictions,
        positive_count
    )

    auc_a = aucs[0]
    auc_b = aucs[1]

    variance_difference = (
        covariance[0, 0]
        + covariance[1, 1]
        - 2 * covariance[0, 1]
    )

    if variance_difference <= 0:

        p_value = 1.0

    else:

        z = (
            abs(auc_a - auc_b)
            / np.sqrt(variance_difference)
        )

        from math import erf, sqrt

        p_value = (
            1.0
            - erf(
                z / sqrt(2.0)
            )
        )

    return (
        auc_a,
        auc_b,
        p_value
    )


# ============================================================
# PAIRED BOOTSTRAP DELTA AUROC
# ============================================================

def paired_bootstrap_delta(
    y_true,
    scores_ref,
    scores_challenger,
    n_bootstrap=N_BOOTSTRAP,
    seed=SEED
):

    y_true = np.asarray(y_true)

    scores_ref = np.asarray(
        scores_ref
    )

    scores_challenger = np.asarray(
        scores_challenger
    )

    rng = np.random.default_rng(seed)

    n = len(y_true)

    delta_point = (
        roc_auc_score(
            y_true,
            scores_ref
        )
        -
        roc_auc_score(
            y_true,
            scores_challenger
        )
    )

    deltas = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_boot = y_true[indices]

        if len(np.unique(y_boot)) < 2:
            continue

        auc_ref = roc_auc_score(
            y_boot,
            scores_ref[indices]
        )

        auc_challenger = roc_auc_score(
            y_boot,
            scores_challenger[indices]
        )

        deltas.append(
            auc_ref - auc_challenger
        )

    deltas = np.asarray(
        deltas
    )

    if len(deltas) == 0:

        raise RuntimeError(
            "Nessun bootstrap valido "
            "per la differenza di AUROC."
        )

    lower = np.percentile(
        deltas,
        100 * ALPHA / 2
    )

    upper = np.percentile(
        deltas,
        100 * (1 - ALPHA / 2)
    )

    return (
        delta_point,
        lower,
        upper
    )


# ============================================================
# FOREST PLOT — AUROC
# ============================================================

def plot_auroc_forest(
    df: pd.DataFrame
):

    df_plot = (
        df
        .set_index("model")
        .reindex(MODEL_ORDER)
        .dropna(subset=["auroc"])
        .reset_index()
    )

    y_pos = np.arange(
        len(df_plot)
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.8)
    )

    # Sfondo alternato
    for i in range(len(df_plot)):

        if i % 2 == 0:

            ax.axhspan(
                i - 0.5,
                i + 0.5,
                color="#f8f9fa",
                zorder=0
            )

    points = df_plot[
        "auroc"
    ].to_numpy()

    lower_errors = (
        points
        - df_plot["auroc_ci_low"].to_numpy()
    )

    upper_errors = (
        df_plot["auroc_ci_high"].to_numpy()
        - points
    )

    for i, model in enumerate(
        df_plot["model"]
    ):

        color = MODEL_COLORS.get(
            model,
            "#333333"
        )

        ax.errorbar(
            points[i],
            i,
            xerr=[
                [lower_errors[i]],
                [upper_errors[i]]
            ],
            fmt="D",
            markersize=9,
            capsize=6,
            capthick=2,
            elinewidth=2,
            color=color,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=1
        )

        ax.text(
            max(
                points
            )
            + max(
                upper_errors
            )
            + 0.015,
            i,
            (
                f"{points[i]:.3f} "
                f"["
                f"{df_plot.loc[i, 'auroc_ci_low']:.3f}, "
                f"{df_plot.loc[i, 'auroc_ci_high']:.3f}"
                f"]"
            ),
            va="center",
            fontsize=10
        )

    ax.set_yticks(y_pos)

    ax.set_yticklabels(
        df_plot["model"],
        fontweight="bold"
    )

    for label in ax.get_yticklabels():

        label.set_color(
            MODEL_COLORS.get(
                label.get_text(),
                "#333333"
            )
        )

    ax.set_xlabel(
        "Patient-level AUROC",
        fontweight="bold"
    )

    ax.set_title(
        "Patient-Level AUROC with 95% Confidence Intervals",
        fontweight="bold",
        pad=15
    )

    min_ci = df_plot[
        "auroc_ci_low"
    ].min()

    max_ci = df_plot[
        "auroc_ci_high"
    ].max()

    ax.set_xlim(
        max(0, min_ci - 0.05),
        min(1.0, max_ci + 0.15)
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )

    ax.grid(
        axis="y",
        visible=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = (
        OUT_DIR
        / "patient_level_auroc_forest.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Salvato: {output_path}"
    )


# ============================================================
# FOREST PLOT — PAIRWISE DELTA AUROC
# ============================================================

def plot_pairwise_forest(
    df_pair: pd.DataFrame
):

    y_pos = np.arange(
        len(df_pair)
    )

    fig, ax = plt.subplots(
        figsize=(9, 4.2)
    )

    # Sfondo alternato
    for i in range(len(df_pair)):

        if i % 2 == 0:

            ax.axhspan(
                i - 0.5,
                i + 0.5,
                color="#f8f9fa",
                zorder=0
            )

    # Linea di assenza di differenza
    ax.axvline(
        0,
        linestyle="--",
        color="#333333",
        linewidth=1.5,
        zorder=1
    )

    points = df_pair[
        "Delta_AUROC"
    ].to_numpy()

    lower_errors = (
        points
        - df_pair["CI_Lower"].to_numpy()
    )

    upper_errors = (
        df_pair["CI_Upper"].to_numpy()
        - points
    )

    for i, model in enumerate(
        df_pair["Challenger"]
    ):

        color = MODEL_COLORS.get(
            model,
            "#333333"
        )

        ax.errorbar(
            points[i],
            i,
            xerr=[
                [lower_errors[i]],
                [upper_errors[i]]
            ],
            fmt="s",
            markersize=9,
            capsize=6,
            capthick=2,
            elinewidth=2,
            color=color,
            zorder=3,
            markeredgecolor="white",
            markeredgewidth=1
        )

        delta = points[i]

        low = df_pair.loc[
            i,
            "CI_Lower"
        ]

        high = df_pair.loc[
            i,
            "CI_Upper"
        ]

        p_value = df_pair.loc[
            i,
            "p_value"
        ]

        if p_value < 0.001:

            p_text = "< 0.001"

        else:

            p_text = f"= {p_value:.3f}"

        ax.text(
            max(
                df_pair["CI_Upper"].max(),
                0
            ) + 0.01,
            i,
            (
                f"Δ={delta:.3f} "
                f"[{low:.3f}, {high:.3f}] "
                f"| p {p_text}"
            ),
            va="center",
            fontsize=10
        )

    ax.set_yticks(y_pos)

    ax.set_yticklabels(
        df_pair["Challenger"],
        fontweight="bold"
    )

    for label in ax.get_yticklabels():

        label.set_color(
            MODEL_COLORS.get(
                label.get_text(),
                "#333333"
            )
        )

    ax.set_xlabel(
        "Δ AUROC "
        "(PatchCore − Challenger)",
        fontweight="bold"
    )

    ax.set_title(
        "Paired Comparison of Patient-Level AUROC",
        fontweight="bold",
        pad=15
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.5
    )

    ax.grid(
        axis="y",
        visible=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    output_path = (
        OUT_DIR
        / "patient_level_pairwise_auroc.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"  ✓ Salvato: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def run():

    print("=" * 70)
    print("PATIENT-LEVEL STATISTICAL ANALYSIS")
    print("=" * 70)

    print(
        f"\nBootstrap: {N_BOOTSTRAP} resampling"
    )

    print(
        f"Confidence level: "
        f"{100 * (1 - ALPHA):.0f}%"
    )

    print(
        f"Reference model: "
        f"{REFERENCE_MODEL_NAME}"
    )

    print(
        f"Unità statistica: PAZIENTE"
    )


    # ========================================================
    # LOAD + PATIENT AGGREGATION
    # ========================================================

    model_data = {}

    for model_name, relative_path in SOURCES:

        csv_path = (
            RESULTS_DIR
            / relative_path
        )

        if not csv_path.exists():

            print(
                f"\n[SKIP] {model_name}: "
                f"file non trovato."
            )

            continue

        df = pd.read_csv(
            csv_path
        )

        required = {
            "patient_id",
            "true_label",
            "anomaly_score"
        }

        missing = (
            required
            - set(df.columns)
        )

        if missing:

            print(
                f"\n[SKIP] {model_name}: "
                f"colonne mancanti {missing}"
            )

            continue

        patient_df = compute_patient_level(
            df
        )

        model_data[
            model_name
        ] = patient_df

        print(
            f"\n[OK] {model_name}"
        )

        print(
            f"     Slice:   {len(df)}"
        )

        print(
            f"     Pazienti: {len(patient_df)}"
        )

        print(
            f"     Positivi: "
            f"{int(patient_df['patient_label'].sum())}"
        )

        print(
            f"     Negativi: "
            f"{int((patient_df['patient_label'] == 0).sum())}"
        )


    if not model_data:

        print(
            "\nNessun modello disponibile."
        )

        return


    # ========================================================
    # BOOTSTRAP AUROC + AP
    # ========================================================

    print("\n")
    print("=" * 70)
    print("BOOTSTRAP PATIENT-LEVEL METRICS")
    print("=" * 70)

    rows = []

    for model_name in MODEL_ORDER:

        if model_name not in model_data:
            continue

        df = model_data[
            model_name
        ]

        y_true = df[
            "patient_label"
        ].to_numpy()

        scores = df[
            "patient_score"
        ].to_numpy()

        if len(np.unique(y_true)) < 2:

            print(
                f"[SKIP] {model_name}: "
                "una sola classe."
            )

            continue

        auroc, auroc_lo, auroc_hi = (
            bootstrap_metric(
                y_true,
                scores,
                roc_auc_score
            )
        )

        ap, ap_lo, ap_hi = (
            bootstrap_metric(
                y_true,
                scores,
                average_precision_score
            )
        )

        rows.append({
            "model": model_name,
            "n_patients": len(df),

            "auroc": auroc,
            "auroc_ci_low": auroc_lo,
            "auroc_ci_high": auroc_hi,

            "ap": ap,
            "ap_ci_low": ap_lo,
            "ap_ci_high": ap_hi,
        })

        print(
            f"\n{model_name}"
        )

        print(
            f"  AUROC = {auroc:.4f} "
            f"[{auroc_lo:.4f}, {auroc_hi:.4f}]"
        )

        print(
            f"  AP    = {ap:.4f} "
            f"[{ap_lo:.4f}, {ap_hi:.4f}]"
        )


    result_df = pd.DataFrame(
        rows
    )

    if result_df.empty:

        print(
            "\nNessun risultato statistico."
        )

        return


    # ========================================================
    # SAVE BOOTSTRAP RESULTS
    # ========================================================

    bootstrap_path = (
        OUT_DIR
        / "patient_level_bootstrap.csv"
    )

    result_df.to_csv(
        bootstrap_path,
        index=False
    )

    print(
        f"\n✓ Salvato: {bootstrap_path}"
    )


    # ========================================================
    # AUROC FOREST PLOT
    # ========================================================

    plot_auroc_forest(
        result_df
    )


    # ========================================================
    # PAIRWISE DELONG + PAIRED BOOTSTRAP
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "PAIRWISE PATIENT-LEVEL AUROC TEST"
    )
    print("=" * 70)


    if REFERENCE_MODEL_NAME not in model_data:

        print(
            "\nReference model non disponibile."
        )

        return


    reference_df = model_data[
        REFERENCE_MODEL_NAME
    ]


    pairwise_results = []


    for challenger_name in MODEL_ORDER:

        if (
            challenger_name
            == REFERENCE_MODEL_NAME
        ):
            continue

        if challenger_name not in model_data:
            continue


        challenger_df = model_data[
            challenger_name
        ]


        # ----------------------------------------------------
        # MATCHING DEI PAZIENTI
        # ----------------------------------------------------

        merged = pd.merge(
            reference_df[
                [
                    "patient_id",
                    "patient_label",
                    "patient_score"
                ]
            ],
            challenger_df[
                [
                    "patient_id",
                    "patient_label",
                    "patient_score"
                ]
            ],
            on="patient_id",
            suffixes=(
                "_ref",
                "_chal"
            )
        )


        # ----------------------------------------------------
        # VERIFICA LABEL
        # ----------------------------------------------------

        labels_match = np.array_equal(
            merged["patient_label_ref"].to_numpy(),
            merged["patient_label_chal"].to_numpy()
        )


        if not labels_match:

            print(
                f"\n[SKIP] {challenger_name}: "
                "le patient labels non coincidono."
            )

            continue


        if len(merged) < 2:

            print(
                f"\n[SKIP] {challenger_name}: "
                "troppi pochi pazienti condivisi."
            )

            continue


        y_true = merged[
            "patient_label_ref"
        ].to_numpy()

        scores_ref = merged[
            "patient_score_ref"
        ].to_numpy()

        scores_chal = merged[
            "patient_score_chal"
        ].to_numpy()


        if len(np.unique(y_true)) < 2:

            print(
                f"\n[SKIP] {challenger_name}: "
                "una sola classe."
            )

            continue


        # ----------------------------------------------------
        # DELONG
        # ----------------------------------------------------

        auc_ref, auc_chal, p_value = (
            delong_test(
                y_true,
                scores_ref,
                scores_chal
            )
        )


        # ----------------------------------------------------
        # PAIRED BOOTSTRAP DELTA
        # ----------------------------------------------------

        delta, ci_low, ci_high = (
            paired_bootstrap_delta(
                y_true,
                scores_ref,
                scores_chal
            )
        )


        significant = (
            "SI"
            if p_value < ALPHA
            else "NO"
        )


        pairwise_results.append({

            "Reference":
                REFERENCE_MODEL_NAME,

            "Challenger":
                challenger_name,

            "n_patients":
                len(merged),

            "AUROC_Reference":
                auc_ref,

            "AUROC_Challenger":
                auc_chal,

            "Delta_AUROC":
                delta,

            "CI_Lower":
                ci_low,

            "CI_Upper":
                ci_high,

            "p_value":
                p_value,

            "Significant":
                significant
        })


        print(
            f"\nPatchCore vs "
            f"{challenger_name}"
        )

        print(
            f"  n patients = {len(merged)}"
        )

        print(
            f"  AUROC PatchCore = "
            f"{auc_ref:.4f}"
        )

        print(
            f"  AUROC Challenger = "
            f"{auc_chal:.4f}"
        )

        print(
            f"  Δ AUROC = "
            f"{delta:.4f}"
        )

        print(
            f"  IC95% = "
            f"[{ci_low:.4f}, {ci_high:.4f}]"
        )

        print(
            f"  DeLong p = "
            f"{p_value:.6f}"
        )

        print(
            f"  Significativo (α={ALPHA}) = "
            f"{significant}"
        )


    # ========================================================
    # SAVE PAIRWISE RESULTS
    # ========================================================

    if not pairwise_results:

        print(
            "\nNessun confronto pairwise disponibile."
        )

        return


    pairwise_df = pd.DataFrame(
        pairwise_results
    )


    pairwise_path = (
        OUT_DIR
        / "patient_level_pairwise_delong.csv"
    )


    pairwise_df.to_csv(
        pairwise_path,
        index=False
    )


    print(
        f"\n✓ Salvato: {pairwise_path}"
    )


    # ========================================================
    # PAIRWISE FOREST PLOT
    # ========================================================

    plot_pairwise_forest(
        pairwise_df
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PATIENT-LEVEL STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        "\nOutput:"
    )

    print(
        f"  {bootstrap_path}"
    )

    print(
        f"  {OUT_DIR / 'patient_level_auroc_forest.png'}"
    )

    print(
        f"  {pairwise_path}"
    )

    print(
        f"  {OUT_DIR / 'patient_level_pairwise_auroc.png'}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run()'''