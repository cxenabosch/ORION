#!/usr/bin/env python3
import os
from pathlib import Path
import subprocess

# --- Docker Configuration ---
IMAGES_TS_DIR = Path("/app/imagesTs")
OUTPUT_DIR = Path("/output")

def main():
    print(">>> Initializing ORION deep learning inference...")

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["nnUNet_results"] = "/weights"

    cmd_predict = [
        "nnUNetv2_predict",
        "-i", str(IMAGES_TS_DIR),
        "-o", str(OUTPUT_DIR),
        "-d", "1",
        "-c", "3d_fullres",
        "-tr", "nnUNetTrainer",
        "-f", "all",
        "-chk", "checkpoint_best.pth",
        "--save_probabilities",
        "-device", "cpu"
    ]

    # Silence standard nnU-Net verbosity unless there's an error
    subprocess.run(cmd_predict, check=True, env=env, stdout=subprocess.DEVNULL)

    print(">>> Applying nomenclature to output segmentations...")
    for file_path in OUTPUT_DIR.iterdir():
        if file_path.is_file() and not file_path.name.startswith("orion_"):
            new_name = "orion_" + file_path.name
            new_path = OUTPUT_DIR / new_name
            file_path.rename(new_path)

    print(">>> ORION pipeline finished successfully! Results are in your output folder.")

if __name__ == "__main__":
    main()
