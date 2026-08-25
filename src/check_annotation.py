import os

root = "BraTS2021/test"

for folder in ["normal", "tumor", "annotation"]:
    path = os.path.join(root, folder)

    print(f"\n--- {folder} ---")

    if os.path.exists(path):
        files = os.listdir(path)
        print("Numero file:", len(files))
        print("Primi 20:")
        for f in files[:20]:
            print(" ", f)
    else:
        print("CARTELLA NON TROVATA")