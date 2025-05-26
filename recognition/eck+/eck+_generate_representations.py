import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import yaml
import sys

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

def extract_crop_size_from_filename(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split('_')
    w = int([p for p in parts if p.startswith('w')][0][1:])
    h = int([p for p in parts if p.startswith('h')][0][1:])
    return (h, w)

def save_image(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in [1, 3]:
        array = np.transpose(array, (1, 2, 0))
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)

def pad_to_square(img, rep):
    img = np.squeeze(img)
    h, w = img.shape[:2]
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2

    if img.ndim == 2:
        fondo = 0 if rep == "sae" else 127 if rep == "event_accumulate" else 0
        padded = np.full((size, size), fondo, dtype=np.uint8)
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = img
    elif img.ndim == 3:
        fondo = [255, 255, 255] if rep == "tencode" else [0, 0, 0]
        padded = np.full((size, size, 3), fondo, dtype=np.uint8)
        padded[pad_y:pad_y + h, pad_x:pad_x + w] = img
    else:
        raise ValueError(f"[ERROR] Imagen con forma no soportada: {img.shape}")

    # Redimensionar a tamaño final
    return np.array(Image.fromarray(padded).resize(IMG_SIZE, Image.BILINEAR))

def process_all():
    for split in ['Train_Set', 'Test_Set']:
        print(f"\nProcesando {split}...")
        for class_id in sorted(os.listdir(os.path.join(INPUT_DIR, split))):
            class_path = os.path.join(INPUT_DIR, split, class_id)
            for sample_file in sorted(os.listdir(class_path)):
                if not sample_file.endswith('.npy'):
                    continue

                sample_path = os.path.join(class_path, sample_file)

                events = np.load(sample_path)
                if len(events) < 2:
                    continue

                events[:, 0] = events[:, 0].astype(np.int32)
                events[:, 1] = events[:, 1].astype(np.int32)
                events[:, 2] = (events[:, 2] * 1e6).astype(np.uint64)  # s → µs

                t_min = events[:, 2].min()
                t_max = events[:, 2].max()

                start_times = np.arange(t_min, t_max - TIME_WINDOW + 1, OFFSET_TIME_WINDOW)

                crop_size = extract_crop_size_from_filename(sample_file)

                for t_start in start_times:
                    t_end = t_start + TIME_WINDOW
                    window_events = events[(events[:, 2] >= t_start) & (events[:, 2] < t_end)].copy()

                    mask_space = (
                        (window_events[:, 0] >= 0) & (window_events[:, 0] < crop_size[1]) &
                        (window_events[:, 1] >= 0) & (window_events[:, 1] < crop_size[0])
                    )
                    window_events = window_events[mask_space]

                    if len(window_events) < 2:
                        continue

                    # Asegurar tipos consistentes
                    x = window_events[:, 0].astype(np.uint16)
                    y = window_events[:, 1].astype(np.uint16)
                    t = window_events[:, 2].astype(np.uint64)
                    p = (window_events[:, 3] > 0).astype(np.uint8)

                    window_events = np.stack([x, y, t, p], axis=-1)

                    #print(window_events)

                    if len(window_events) < 2:
                        continue

                    tag = f"{os.path.splitext(sample_file)[0]}_t{int((t_start - t_min) / 1000):04d}"

                    for rep in REPRESENTATIONS:
                        rep_dir = os.path.join(OUTPUT_DIR, rep, split, class_id)
                        os.makedirs(rep_dir, exist_ok=True)
                        out_path = os.path.join(rep_dir, f"{tag}.png")

                        if rep == "event_accumulate":
                            img = make_event_accumulate(window_events, crop_size)
                        elif rep == "sae":
                            img = make_sae(window_events, crop_size)
                        elif rep == "tbr":
                            img = make_tbr(window_events, crop_size, num_bins=TBR_BINS)
                        elif rep == "tbr_tensor":
                            tensor = make_tbr_tensor(window_events, crop_size, IMG_SIZE, TBR_BINS, rescale=True)
                            np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                            img = None
                        elif rep == "tqr_tensor":
                            tensor = make_tqr_tensor(window_events, crop_size, IMG_SIZE, TBR_BINS, rescale=True)
                            np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                            img = None
                        elif rep == "tencode":
                            img = make_tencode(window_events, crop_size)
                        elif rep == "behi":
                            img = make_behi(window_events, crop_size)
                        else:
                            img = None

                        if img is not None:
                            padded = pad_to_square(img, rep)
                            if padded.ndim == 3:
                                save_color_image(padded, out_path)
                            else:
                                save_image(padded, out_path)

if __name__ == "__main__":
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="E-CK+ representation generator")
    parser.add_argument("--input-dir", type=str, default=f"{INPUT_ROOT}/eck+/", help="Path to the cropped event files (.npy)")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_ROOT}/eck+/", help="Path where to save the representations")
    args = parser.parse_args()

    IMG_SIZE = (260, 260)
    TBR_BINS = 8
    TIME_WINDOW = np.uint64(100_000)   # 100 ms en µs
    OFFSET_TIME_WINDOW = np.uint64(10_000)  # 50 ms en µs
    INPUT_DIR = args.input_dir + "ddbb_cropped/e-ck+_346_full_cropped_events_100ms"
    OUTPUT_DIR = args.output_dir + f"e-ck+_rep_{int(TIME_WINDOW / 1000)}ms"
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]

    process_all()
