import os
import numpy as np
import glob
import random
from tqdm import tqdm
from PIL import Image
import cv2
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

def read_nmnist_bin(filepath):
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if len(data) % 5 != 0:
        raise ValueError(f"Archivo {filepath} no tiene múltiplo de 5 bytes")

    data = data.reshape(-1, 5)
    x = data[:, 0].astype(np.uint16)
    y = data[:, 1].astype(np.uint16)
    polarity = ((data[:, 2] & 0x80) >> 7).astype(np.uint8)
    t_high = (data[:, 2] & 0x7F).astype(np.uint32)
    t_mid = data[:, 3].astype(np.uint32)
    t_low = data[:, 4].astype(np.uint32)
    timestamp = (t_high << 16) | (t_mid << 8) | t_low

    return np.stack([x, y, timestamp, polarity], axis=-1)  # ← tipo ndarray, shape (N, 4)


def save_image(array, path):
    array = np.squeeze(array)
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    Image.fromarray(array.astype(np.uint8)).save(path)


def main():
    save_dataset_config()
    split_log = []

    for split in SPLIT:
        for label in map(str, range(10)):
            files = sorted(glob.glob(os.path.join(INPUT_DIR, split, label, "*.bin")))
            print(f"[INFO] {split}/{label}: {len(files)} archivos")

            for path_bin in tqdm(files, desc=f"{split}/{label}"):
                fname = os.path.splitext(os.path.basename(path_bin))[0]

                try:
                    events = read_nmnist_bin(path_bin)
                    if len(events) < 2:
                        continue

                    t_min = events[:, 2].min()
                    t_max = events[:, 2].max()
                    duration = t_max - t_min
                    if duration < TIME_WINDOW:
                        continue

                    # Cortar solo los primeros TIME_WINDOW desde el inicio
                    t_start = t_min
                    t_end = t_start + TIME_WINDOW
                    window = events[(events[:, 2] >= t_start) & (events[:, 2] < t_end)]

                    if len(window) < 2:
                        continue

                    tag = f"{fname}_t0"
                    split_log.append(f"{split} {label}/{tag}.bin\n")

                    for rep in REPRESENTATIONS:
                        rep_dir = os.path.join(OUTPUT_DIR, rep, split, label)
                        os.makedirs(rep_dir, exist_ok=True)
                        out_path = os.path.join(rep_dir, f"{tag}.png")

                        if rep == "event_accumulate":
                            img = make_event_accumulate(window, IMG_SIZE)
                        elif rep == "sae":
                            img = make_sae(window, IMG_SIZE)
                        elif rep == "tbr":
                            img = make_tbr(window, IMG_SIZE, TBR_BINS)
                        elif rep == "tbr_tensor":
                            tensor = make_tbr_tensor(window, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                            np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                            img = None
                        elif rep == "tqr_tensor":
                            tensor = make_tqr_tensor(window, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                            np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                            img = None
                        elif rep == "tencode":
                            img = make_tencode(window, IMG_SIZE)
                        elif rep == "behi":
                            img = make_behi(window, IMG_SIZE)
                        else:
                            img = None

                        if img is not None:
                            if img.ndim == 3:
                                save_color_image(img, out_path)
                            else:
                                save_image(img, out_path)

                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")

    with open(os.path.join(OUTPUT_DIR, "split_log.txt"), "w") as f:
        f.writelines(split_log)

# === Parámetros ===
if __name__ == "__main__":

    with open("../config_dgx-1.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="N-MNIST representation generator")
    parser.add_argument("--input-dir", type=str, default=f"{INPUT_ROOT}", help="Path to input dir")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_ROOT}", help="Path to output dir")
    args = parser.parse_args()

    IMG_SIZE = (34, 34)
    TBR_BINS = 16
    TIME_WINDOW = np.uint64(100_000)   
    INPUT_DIR = os.path.join(args.input_dir, "nmnist_bin")
    OUTPUT_DIR = os.path.join(args.output_dir, f"nmnist_rep_{int(TIME_WINDOW/1e3)}ms")
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]
    SPLIT = ["Train", "Test"]

    main()