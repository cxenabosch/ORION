#!/usr/bin/env python3
import os
import sys

# --- CONFIGURACIÓ CRÍTICA DE DISPOSITIU (A dalt de tot, abans de nnUNet) ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # Desactiva completament CUDA
os.environ["nnUNet_compile"] = "False"        # Evita intents de compilació per GPU
# --------------------------------------------------------------------------

# Forcem el path per si de cas, encara que ja ho hem arreglat al Dockerfile
sys.path.append("/usr/local/lib/python3.10/dist-packages")

from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import predict_entry_point

# --- Configuració ---
IMAGES_TS_DIR = Path("/app/imagesTs")
OUTPUT_DIR = Path("/output")
RESULTS_DIR = "/weights"

def main():
    print(">>> Initializing ORION deep learning inference...")

    # 1. Configurar variables d'entorn restants
    os.environ["nnUNet_results"] = RESULTS_DIR
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Canviat a 1 per si PyTorch vol utilitzar l'arquitectura del Mac

    # 2. Construir els arguments tal com els espera nnUNet
    args = [
        "-i", str(IMAGES_TS_DIR),
        "-o", str(OUTPUT_DIR),
        "-d", "1",
        "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-f", "all",
        "-chk", "checkpoint_best.pth",
        "-device", "cpu",              # <--- NOU: Li diem a nnU-Net que usi estrictament la CPU
        "--save_probabilities"
    ]

    print(f">>> Arguments passed to nnUNet: {args}")

    # 3. Executar la inferència
    try:
        # Enganyem el nnUNet substituint els arguments del sistema
        sys.argv = ["nnUNet_predict"] + args
        predict_entry_point()
    except Exception as e:
        print(f"!!! Error durant la inferència: {e}")
        exit(1)

    # 4. Post-processament
    print("\n>>> Applying nomenclature to output segmentations...")
    for file_path in OUTPUT_DIR.iterdir():
        if file_path.is_file() and not file_path.name.startswith("orion_"):
            new_name = "orion_" + file_path.name
            file_path.rename(OUTPUT_DIR / new_name)

    print(">>> ORION pipeline finished successfully!")

if __name__ == "__main__":
    main()
