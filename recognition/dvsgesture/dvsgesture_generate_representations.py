import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import glob
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

def save_image(array, path):
    array = np.squeeze(array)
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    Image.fromarray(array).save(path)

def save_dataset_config():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/dataset_config.txt"
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

def process_file(file_path, split, label):
    try:
        events = np.load(file_path, allow_pickle=True)
        x = events[:, 0].astype(np.uint16)
        y = events[:, 1].astype(np.uint16)
        p = events[:, 2].astype(np.uint8)
        t = (events[:, 3] * 1_000).astype(np.uint64)  # ms → µs
        event_array = np.stack([x, y, t, p], axis=1)

        t_min = event_array[:, 2].min()
        t_max = event_array[:, 2].max()
        start_times = np.arange(t_min, t_max - TIME_WINDOW + 1, TIME_WINDOW, dtype=np.uint64)

        # Obtener nombre de carpeta y archivo base
        parent_folder = os.path.basename(os.path.dirname(file_path))
        file_base = os.path.splitext(os.path.basename(file_path))[0]

        for i, t_start in enumerate(start_times):
            t_end = t_start + TIME_WINDOW
            window_events = event_array[(event_array[:, 2] >= t_start) & (event_array[:, 2] < t_end)]

            if len(window_events) == 0:
                continue

            # Tag robusto para evitar colisiones
            tag = f"{parent_folder}_{file_base}_t{i:03d}"

            for rep in REPRESENTATIONS:
                rep_dir = os.path.join(OUTPUT_DIR, rep, split, str(label))
                os.makedirs(rep_dir, exist_ok=True)
                out_path = os.path.join(rep_dir, f"{tag}.png")

                if rep == "event_accumulate":
                    img = make_event_accumulate(window_events, IMG_SIZE)
                    save_image(img, out_path)
                elif rep == "sae":
                    img = make_sae(window_events, IMG_SIZE)
                    save_image(img, out_path)
                elif rep == "tbr":
                    img = make_tbr(window_events, IMG_SIZE, TBR_BINS)
                    save_image(img, out_path)
                elif rep == "tbr_tensor":
                    tensor = make_tbr_tensor(window_events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                    np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                elif rep == "tqr_tensor":
                    tensor = make_tqr_tensor(window_events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                    np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                elif rep == "tencode":
                    img = make_tencode(window_events, IMG_SIZE)
                    save_color_image(img, out_path)
                elif rep == "behi":
                    img = make_behi(window_events, IMG_SIZE)
                    save_image(img, out_path)

    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")


def main():
    save_dataset_config()
    for split in ["train", "test"]:
        base_dir = os.path.join(INPUT_DIR, f"ibmGesture{split.capitalize()}")
        user_folders = sorted(os.listdir(base_dir))

        for user_folder in tqdm(user_folders, desc=f"Procesando {split}"):
            folder_path = os.path.join(base_dir, user_folder)
            files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
            for file_path in files:
                label = int(os.path.splitext(os.path.basename(file_path))[0])
                process_file(file_path, split, label)


if __name__ == "__main__":

    with open("../config_dgx-1.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="DVS128 Gesture representation generator")
    parser.add_argument("--input-dir", type=str, default=f"{INPUT_ROOT}", help="Path to input dir")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_ROOT}", help="Path to output dir")
    parser.add_argument("--timewindow-ms", type=int, default=100, help="Window size in ms (default: 100)")

    args = parser.parse_args()

    IMG_SIZE = (128, 128)
    TBR_BINS = 8
    TIME_WINDOW = np.uint64(args.timewindow_ms * 1_000) # ms -> us     
    
    INPUT_DIR = os.path.join(args.input_dir, "dvsgesture_npy")
    OUTPUT_DIR = os.path.join(args.output_dir, f"dvsgesture_rep_{int(TIME_WINDOW/1e3)}ms")
    
    #REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]
    REPRESENTATIONS = ["event_accumulate", "tbr", "tbr_tensor", "tqr_tensor", "tencode"]

    main()
