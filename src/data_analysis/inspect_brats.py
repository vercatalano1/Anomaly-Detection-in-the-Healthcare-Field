from pathlib import Path
from collections import Counter
import numpy as np
import sys

# ============================================================
# CONFIGURAZIONE
# ============================================================

# Percorso del dataset
DATASET_DIR = Path(
    r"C:\Users\catal\OneDrive\Desktop\Anomaly-Detection-in-the-Healthcare-Field\BraTS2021"
)

# File di output
OUTPUT_DIR = Path("results")
OUTPUT_FILE = OUTPUT_DIR / "inspect.txt"

# Numero massimo di file da analizzare per ogni cartella
MAX_FILES_PER_FOLDER = 10


# ============================================================
# SETUP OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Tee:
    """
    Scrive contemporaneamente sul terminale e su un file.
    """

    def __init__(self, file):
        self.terminal = sys.stdout
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


output_file = open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
)

sys.stdout = Tee(output_file)


# ============================================================
# FUNZIONI UTILI
# ============================================================

def format_bytes(size):
    """Converte bytes in una rappresentazione leggibile."""

    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def inspect_numpy_file(file_path):
    """Analizza un file NumPy (.npy/.npz)."""

    try:

        if file_path.suffix.lower() == ".npy":

            data = np.load(
                file_path,
                allow_pickle=False
            )

            print(f"    shape       : {data.shape}")
            print(f"    dtype       : {data.dtype}")
            print(f"    min         : {np.min(data):.6f}")
            print(f"    max         : {np.max(data):.6f}")
            print(f"    mean        : {np.mean(data):.6f}")
            print(f"    std         : {np.std(data):.6f}")
            print(f"    NaN         : {np.isnan(data).sum()}")
            print(f"    Inf         : {np.isinf(data).sum()}")

        elif file_path.suffix.lower() == ".npz":

            data = np.load(
                file_path,
                allow_pickle=False
            )

            print(f"    arrays      : {list(data.keys())}")

            for key in data.files:

                arr = data[key]

                print(f"\n    [{key}]")
                print(f"      shape     : {arr.shape}")
                print(f"      dtype     : {arr.dtype}")
                print(f"      min       : {np.min(arr):.6f}")
                print(f"      max       : {np.max(arr):.6f}")
                print(f"      mean      : {np.mean(arr):.6f}")
                print(f"      std       : {np.std(arr):.6f}")
                print(f"      NaN       : {np.isnan(arr).sum()}")
                print(f"      Inf       : {np.isinf(arr).sum()}")

    except Exception as e:

        print(
            f"    ERRORE nella lettura: {e}"
        )


def inspect_image_file(file_path):
    """Analizza un'immagine utilizzando PIL."""

    try:

        from PIL import Image

        with Image.open(file_path) as img:

            print(f"    size        : {img.size}")
            print(f"    mode        : {img.mode}")
            print(f"    format      : {img.format}")

            arr = np.asarray(img)

            print(f"    shape       : {arr.shape}")
            print(f"    dtype       : {arr.dtype}")
            print(f"    min         : {arr.min()}")
            print(f"    max         : {arr.max()}")
            print(f"    mean        : {arr.mean():.6f}")
            print(f"    std         : {arr.std():.6f}")

            print(
                f"    NaN         : "
                f"{np.isnan(arr).sum()}"
            )

            print(
                f"    Inf         : "
                f"{np.isinf(arr).sum()}"
            )

    except ImportError:

        print(
            "    PIL non installato. "
            "Installa con: pip install pillow"
        )

    except Exception as e:

        print(
            f"    ERRORE nella lettura: {e}"
        )


# ============================================================
# CONTROLLO DATASET
# ============================================================

if not DATASET_DIR.exists():

    print("=" * 70)
    print("ERRORE")
    print("=" * 70)

    print(
        f"La cartella non esiste: "
        f"{DATASET_DIR.resolve()}"
    )

    print()
    print(
        "Modifica DATASET_DIR all'inizio dello script."
    )

    output_file.close()
    sys.stdout = sys.__stdout__

    exit()


print("=" * 70)
print("BRATS2021 - DATASET INSPECTION")
print("=" * 70)

print(
    f"\nDataset path:\n"
    f"{DATASET_DIR.resolve()}"
)

print(
    f"\nOutput salvato in:\n"
    f"{OUTPUT_FILE.resolve()}"
)


# ============================================================
# 1. STRUTTURA CARTELLE
# ============================================================

print("\n" + "=" * 70)
print("1. STRUTTURA DEL DATASET")
print("=" * 70)

directories = sorted(
    [
        p
        for p in DATASET_DIR.rglob("*")
        if p.is_dir()
    ]
)

print(
    f"\nNumero totale di cartelle: "
    f"{len(directories)}"
)

for directory in directories:

    relative = directory.relative_to(
        DATASET_DIR
    )

    print(f"  {relative}")


# ============================================================
# 2. FILE ED ESTENSIONI
# ============================================================

print("\n" + "=" * 70)
print("2. FILE ED ESTENSIONI")
print("=" * 70)

files = [
    p
    for p in DATASET_DIR.rglob("*")
    if p.is_file()
]

print(
    f"\nNumero totale di file: "
    f"{len(files)}"
)

extensions = Counter(
    p.suffix.lower()
    if p.suffix
    else "[nessuna estensione]"
    for p in files
)

print("\nDistribuzione delle estensioni:")

for extension, count in extensions.most_common():

    print(
        f"  {extension:15s} : {count}"
    )


# ============================================================
# 3. DIMENSIONE DEL DATASET
# ============================================================

print("\n" + "=" * 70)
print("3. DIMENSIONE DEL DATASET")
print("=" * 70)

total_size = sum(
    p.stat().st_size
    for p in files
)

print(
    f"\nDimensione totale: "
    f"{format_bytes(total_size)}"
)


# ============================================================
# 4. FILE PER CARTELLA
# ============================================================

print("\n" + "=" * 70)
print("4. NUMERO DI FILE PER CARTELLA")
print("=" * 70)

for directory in [DATASET_DIR] + directories:

    folder_files = [
        p
        for p in directory.iterdir()
        if p.is_file()
    ]

    if folder_files:

        relative = directory.relative_to(
            DATASET_DIR
        )

        print(
            f"{str(relative):40s} : "
            f"{len(folder_files)} file"
        )


# ============================================================
# 5. ANALISI DI ALCUNI FILE
# ============================================================

print("\n" + "=" * 70)
print("5. ANALISI DEI FILE")
print("=" * 70)

print(
    f"\nVerranno analizzati al massimo "
    f"{MAX_FILES_PER_FOLDER} file per cartella."
)

for directory in [DATASET_DIR] + directories:

    folder_files = sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
        ]
    )

    if not folder_files:
        continue

    relative = directory.relative_to(
        DATASET_DIR
    )

    print("\n" + "-" * 70)
    print(f"CARTELLA: {relative}")
    print("-" * 70)

    selected_files = folder_files[
        :MAX_FILES_PER_FOLDER
    ]

    for file_path in selected_files:

        print(
            f"\n  FILE: {file_path.name}"
        )

        print(
            f"    dimensione   : "
            f"{format_bytes(file_path.stat().st_size)}"
        )

        suffix = file_path.suffix.lower()

        if suffix in [".npy", ".npz"]:

            inspect_numpy_file(
                file_path
            )

        elif suffix in [
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff"
        ]:

            inspect_image_file(
                file_path
            )

        else:

            print(
                f"    tipo         : "
                f"non analizzato automaticamente "
                f"({suffix})"
            )


# ============================================================
# 6. FILE PIÙ GRANDI
# ============================================================

print("\n" + "=" * 70)
print("6. FILE PIÙ GRANDI")
print("=" * 70)

largest_files = sorted(
    files,
    key=lambda p: p.stat().st_size,
    reverse=True
)[:10]

for file_path in largest_files:

    relative = file_path.relative_to(
        DATASET_DIR
    )

    print(
        f"{format_bytes(file_path.stat().st_size):>12s}  "
        f"{relative}"
    )


# ============================================================
# 7. FILE PIÙ PICCOLI
# ============================================================

print("\n" + "=" * 70)
print("7. FILE PIÙ PICCOLI")
print("=" * 70)

smallest_files = sorted(
    files,
    key=lambda p: p.stat().st_size
)[:10]

for file_path in smallest_files:

    relative = file_path.relative_to(
        DATASET_DIR
    )

    print(
        f"{format_bytes(file_path.stat().st_size):>12s}  "
        f"{relative}"
    )


# ============================================================
# FINE
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETATA")
print("=" * 70)

print(
    "\nQuesto script non modifica "
    "alcun file del dataset."
)

print(
    f"\nReport salvato in: "
    f"{OUTPUT_FILE.resolve()}"
)


# ============================================================
# CHIUSURA FILE
# ============================================================

output_file.flush()
output_file.close()

sys.stdout = sys.__stdout__