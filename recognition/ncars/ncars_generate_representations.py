import os
import numpy as np
import glob
import random
from tqdm import tqdm
from PIL import Image
import cv2
from src.io.psee_loader import PSEELoader
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
BASE_DIR = {BASE_DIR}
OUTPUT_DIR = {OUTPUT_DIR}
IMG_SIZE (final) = {IMG_SIZE}
TBR_BINS = {TBR_BINS}
TIME_WINDOW = {int(TIME_WINDOW)} (μs)
REPRESENTATIONS = {', '.join(REPRESENTATIONS)}
        """
    with open(path, "w") as f:
        f.write(config_text)


def pad_to_square(image, background=0):
    """
    Rellena una imagen para que sea cuadrada, agregando fondo (negro=0 o blanco=255).
    Soporta imágenes en escala de grises (2D) y RGB (3D).
    """
    h, w = image.shape[:2]
    size = max(h, w)
    if image.ndim == 2:
        square = np.full((size, size), background, dtype=image.dtype)
    else:
        square = np.full((size, size, image.shape[2]), background, dtype=image.dtype)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = image
    return square


def save_image_from_events(array, path, max_y, max_x, final_size, background=0):
    array = np.squeeze(array)
    padded_size = max(max_y, max_x)

    # Padding centrado
    padded = np.full((padded_size, padded_size), background, dtype=array.dtype)
    y_offset = (padded_size - max_y) // 2
    x_offset = (padded_size - max_x) // 2
    padded[y_offset:y_offset + max_y, x_offset:x_offset + max_x] = array

    # Resize final
    resized = Image.fromarray(padded).resize(final_size, Image.BILINEAR)
    Image.fromarray(np.array(resized)).save(path)


def save_color_image_from_events(array, path, max_y, max_x, final_size, background=255):
    padded_size = max(max_y, max_x)

    # Padding centrado
    padded = np.full((padded_size, padded_size, 3), background, dtype=array.dtype)
    y_offset = (padded_size - max_y) // 2
    x_offset = (padded_size - max_x) // 2
    padded[y_offset:y_offset + max_y, x_offset:x_offset + max_x] = array

    # Resize final
    resized = Image.fromarray(padded).resize(final_size, Image.BILINEAR)
    Image.fromarray(np.array(resized)).save(path)


def parse_legacy_dat_events(filepath):
    video = PSEELoader(filepath)
    video.seek_time(0)
    duration = video.total_time()
    events = video.load_delta_t(duration)

    return np.array([
        (e['x'], e['y'], e['t'], e['p']) for e in events
    ], dtype=[
        ('x', np.uint16),
        ('y', np.uint16),
        ('timestamp', np.uint64),
        ('polarity', np.bool_)
    ])

def main():
    save_dataset_config() 

    max_width = 0
    max_height = 0

    for rep in REPRESENTATIONS:
        for split in ["train", "test"]:
            for label in ["cars", "background"]:
                os.makedirs(os.path.join(OUTPUT_DIR, rep, split, label), exist_ok=True)

    for split in ["train", "test"]:
        for label in ["cars", "background"]:
            folder_path = os.path.join(BASE_DIR, f"n-cars_{split}", label)
            files = sorted(glob.glob(os.path.join(folder_path, "*.dat")))
            print(f"[INFO] Procesando {len(files)} archivos en {folder_path}")

            for file_path in tqdm(files, desc=f"{split}/{label}"):
                fname = os.path.splitext(os.path.basename(file_path))[0]
                try:
                    events = parse_legacy_dat_events(file_path)
                    events = np.stack([events['x'], events['y'], events['timestamp'], events['polarity']], axis=-1)


                    max_x = int(events[:, 0].max()) + 1
                    max_y = int(events[:, 1].max()) + 1

                    if max_x > max_width:
                        max_width = max_x
                    if max_y > max_height:
                        max_height = max_y

                    if len(events) == 0:
                        print(f"[WARNING] Archivo vacío: {file_path}")
                        continue
                except Exception as e:
                    print(f"[ERROR] Fallo leyendo {file_path}: {e}")
                    continue

                for rep in REPRESENTATIONS:
                    
                    out_path = f"{OUTPUT_DIR}/{rep}/{split}/{label}/{fname}.png"

                    if rep == "event_accumulate":
                        img = make_event_accumulate(events, (max_y, max_x))
                        save_image_from_events(img, out_path, max_y, max_x, final_size=IMG_SIZE, background=127)
                    elif rep == "sae":
                        img = make_sae(events, (max_y, max_x))
                        save_image_from_events(img, out_path, max_y, max_x, final_size=IMG_SIZE)
                    elif rep == "tbr":
                        img = make_tbr(events, (max_y, max_x), TBR_BINS)
                        save_image_from_events(img, out_path, max_y, max_x, final_size=IMG_SIZE)
                    elif rep == "tbr_tensor":
                        tensor = make_tbr_tensor(events, original_size=(max_y, max_x), final_size=IMG_SIZE, num_bins=TBR_BINS, rescale=True)
                        npy_path = os.path.join(OUTPUT_DIR, rep, split, label, f"{fname}.npy")
                        np.save(npy_path, tensor)
                    elif rep == "tqr_tensor":
                        tensor = make_tqr_tensor(events, original_size=(max_y, max_x), final_size=IMG_SIZE, num_bins=TBR_BINS, rescale=True)
                        npy_path = os.path.join(OUTPUT_DIR, rep, split, label, f"{fname}.npy")
                        np.save(npy_path, tensor)
                    elif rep == "tencode":
                        img = make_tencode(events, (max_y, max_x))
                        save_color_image_from_events(img, out_path, max_y, max_x, final_size=IMG_SIZE)
                    elif rep == "behi":
                        img = make_behi(events, (max_y, max_x))
                        save_image_from_events(img, out_path, max_y, max_x, final_size=IMG_SIZE)


    print(f"\nMaximum resolution detected among all files:")
    print(f" → Max width (max_x): {max_width}")
    print(f" → Max height (max_y): {max_height}")


if __name__ == "__main__":

    with open("../config_dgx-1.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="N-Cars representation generator")

    parser.add_argument(
        "--input-dir",
        type=str,
        default=f"{INPUT_ROOT}",
        help="Path to the input directory (.dat files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{OUTPUT_ROOT}",
        help="Path to the directory where to save the representations"
    )

    args = parser.parse_args()

    # === Configuración global ===
    IMG_SIZE = (120, 120)
    TBR_BINS = 8
    TIME_WINDOW = 100_000  # en microsegundos
    SPLIT = {"train", "test"}
    BASE_DIR = args.input_dir + "ncars_dat"
    OUTPUT_DIR = args.output_dir + f"ncars_rep_{str(int(TIME_WINDOW/1e3))}ms"
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]

    main()
