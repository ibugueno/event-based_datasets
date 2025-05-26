import os
import numpy as np
import glob
import random
from tqdm import tqdm
from PIL import Image
import cv2
from dv import AedatFile
import sys
import argparse
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import (
    make_event_accumulate,
    make_sae,
    make_tbr,
    make_tbr_tensor,
    make_tqr_tensor,
    make_tencode,
    make_behi
)

def save_dataset_config():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path=f"{OUTPUT_DIR}/dataset_config.txt"
    config_text = f"""# Dataset Parameters
INPUT_DIR = {INPUT_DIR}
OUTPUT_DIR = {OUTPUT_DIR}
IMG_SIZE (final) = {IMG_SIZE}
TBR_BINS = {TBR_BINS}
TIME_WINDOW = {int(TIME_WINDOW)} (μs)
REPRESENTATIONS = {', '.join(REPRESENTATIONS)}
        """
    with open(path, "w") as f:
        f.write(config_text)

def save_image(array, path):
    array = np.squeeze(array)
    array = np.rot90(array, k=1)  # Rotar 90° a la izquierda
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    array = np.rot90(array, k=1)  # Rotar 90° a la izquierda
    Image.fromarray(array).save(path)

def parse_aedat4_events(filepath):
    with AedatFile(filepath) as f:
        # Concatenar todos los paquetes de eventos en una lista
        events = [packet for packet in f['events'].numpy()]
        if not events:
            return np.empty((0, 4), dtype=np.uint16)
        events = np.hstack(events)
        return np.stack([
            events['x'],
            events['y'],
            events['timestamp'],
            events['polarity']
        ], axis=-1)

def main():
    save_dataset_config()  # ← guarda los parámetros al inicio

    for rep in REPRESENTATIONS:
        for split in SPLIT:
            os.makedirs(os.path.join(OUTPUT_DIR, rep, split), exist_ok=True)

    split_log = []

    for class_name in sorted(os.listdir(INPUT_DIR)):
        class_dir = os.path.join(INPUT_DIR, class_name)
        files = sorted(glob.glob(os.path.join(class_dir, "*.aedat4")))
        print(f"[INFO] Encontrados {len(files)} archivos en {class_dir}")
        random.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * SPLIT["train"])
        n_val = int(n_total * SPLIT["val"])
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:]
        }

        for split, file_list in split_files.items():
            for path in tqdm(file_list, desc=f"{class_name} - {split}"):
                fname = os.path.splitext(os.path.basename(path))[0]
                try:
                    events = parse_aedat4_events(path)
                    if len(events) == 0:
                        print(f"[WARNING] Archivo vacío: {path}")
                        continue
                    else:
                        t_max = np.uint64(events[:, 2].max())
                        t_min = t_max - TIME_WINDOW if t_max >= TIME_WINDOW else np.uint64(0)
                        events = events[events[:, 2] >= t_min]

                except Exception as e:
                    print(f"Error parsing {path}: {e}")
                    continue

                for rep in REPRESENTATIONS:
                    rep_dir = os.path.join(OUTPUT_DIR, rep, split, class_name)
                    os.makedirs(rep_dir, exist_ok=True)

                    out_path = os.path.join(rep_dir, f"{fname}.png")

                    if rep == "event_accumulate":
                        img = make_event_accumulate(events, IMG_SIZE)
                        save_image(img, out_path)

                    elif rep == "sae":
                        img = make_sae(events, IMG_SIZE)
                        save_image(img, out_path)
                    
                    elif rep == "tbr":
                        img = make_tbr(events, IMG_SIZE, TBR_BINS)
                        save_image(img, out_path)
                    
                    elif rep == "tbr_tensor":
                        bin_str_array = make_tbr_tensor(events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                        npy_path = os.path.join(rep_dir, f"{fname}.npy")
                        np.save(npy_path, bin_str_array)
                    
                    elif rep == "tqr_tensor":
                        bin_str_array = make_tqr_tensor(events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                        npy_path = os.path.join(rep_dir, f"{fname}.npy")
                        np.save(npy_path, bin_str_array)

                    elif rep == "tencode":
                        img = make_tencode(events, IMG_SIZE)
                        save_color_image(img, out_path)
                    
                    elif rep == "behi":
                        img = make_behi(events, IMG_SIZE)
                        save_image(img, out_path)
                
                split_log.append(f"{split} {class_name}/{fname}.aedat4\n")

    with open(os.path.join(OUTPUT_DIR, "split.txt"), "w") as f:
        f.writelines(split_log)

if __name__ == "__main__":

    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="CIFAR10-DVS representation generator")

    parser.add_argument(
        "--input-dir",
        type=str,
        default=f"{INPUT_ROOT}/cifar10dvs/",
        help="Path to the input directory (.aedat4 files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{OUTPUT_ROOT}/cifar10dvs/",
        help="Path to the directory where to save the representations"
    )

    args = parser.parse_args()

    # === Configuración global ===
    IMG_SIZE = (128, 128)
    TBR_BINS = 8
    TIME_WINDOW = 100_000  # en microsegundos
    SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
    INPUT_DIR = args.input_dir + "cifar10dvs_aedat/train"
    OUTPUT_DIR = args.output_dir + f"/cifar10dvs_rep_{str(int(TIME_WINDOW/1e3))}ms"
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]

    main()
