import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
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

def save_dataset_config():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path=f"{OUTPUT_DIR}/dataset_config.txt"
    config_text = f"""# Dataset Parameters
INPUT_DIR = {INPUT_DIR}
ANNOTATION_DIR = {ANNOTATION_DIR}
OUTPUT_DIR = {OUTPUT_DIR}
IMG_SIZE (final) = {IMG_SIZE}
TBR_BINS = {TBR_BINS}
TIME_WINDOW = {int(TIME_WINDOW)} (μs)
OFFSET_TIME_WINDOW = {int(OFFSET_TIME_WINDOW)} (μs)
REPRESENTATIONS = {', '.join(REPRESENTATIONS)}
SPLIT_RATIO = {SPLIT_RATIO}
        """
    with open(path, "w") as f:
        f.write(config_text)

# === Utilidades ===
def save_image(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.squeeze(array)
    if array.ndim == 3 and array.shape[0] in [1, 3]:  # (C, H, W) → (H, W, C)
        array = np.transpose(array, (1, 2, 0))
    if array.ndim > 3:
        raise ValueError(f"[ERROR] Imagen con forma no soportada: {array.shape}")
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(array).save(path)

def pad_to_square(img, rep):
    img = np.squeeze(img)

    if img.ndim not in [2, 3]:
        raise ValueError(f"[ERROR] Imagen con forma no soportada: {img.shape}")

    h, w = img.shape[:2]
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2

    # === Fondo según representación ===
    if img.ndim == 2:
        fondo = 0 if rep == "sae" else 127 if rep == "event_accumulate" else 0
        padded = np.full((size, size), fondo, dtype=np.uint8)
        padded[pad_y:pad_y+h, pad_x:pad_x+w] = img
    elif img.ndim == 3:
        fondo = [255, 255, 255] if rep == "tencode" else [0, 0, 0]
        padded = np.full((size, size, 3), fondo, dtype=np.uint8)
        padded[pad_y:pad_y+h, pad_x:pad_x+w] = img
    else:
        raise ValueError(f"[ERROR] Imagen con forma no soportada: {img.shape}")

    # === Redimensionar a 180x180
    pil_img = Image.fromarray(padded)
    resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    return np.array(resized)


def read_ncaltech_bin(filepath):
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    data = data.reshape(-1, 5)
    x = data[:, 0].astype(np.uint16)
    y = data[:, 1].astype(np.uint16)
    polarity = ((data[:, 2] & 0x80) >> 7).astype(np.uint8)

    t_high = (data[:, 2] & 0x7F).astype(np.uint64)
    t_low = ((data[:, 3].astype(np.uint64) << 8) | data[:, 4].astype(np.uint64))
    timestamp = (t_high << 16) | t_low

    return np.stack([x, y, timestamp, polarity], axis=-1)


def read_annotation_bbox(path):
    data = np.fromfile(path, dtype=np.int16)
    n_points = data[1]
    points = data[2:2 + 2 * n_points].reshape(n_points, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def crop_events(events, bbox):
    x_min, y_min, x_max, y_max = bbox
    mask = (events[:, 0] >= x_min) & (events[:, 0] <= x_max) & \
           (events[:, 1] >= y_min) & (events[:, 1] <= y_max)
    events = events[mask]
    events[:, 0] -= x_min
    events[:, 1] -= y_min
    return events, (y_max - y_min + 1, x_max - x_min + 1)



# === Main ===
def main():
    save_dataset_config()  

    split_log = []

    total_duration = 0
    valid_samples = 0


    for class_name in ["garfield"]:
        bin_files = sorted(glob.glob(os.path.join(INPUT_DIR, class_name, "*.bin")))
        random.shuffle(bin_files)
        n = len(bin_files)
        if n < 2:
            continue  # saltar si no hay al menos 2 muestras

        split_counts = {
            "train": max(1, int(SPLIT_RATIO["train"] * n)),
        }
        split_counts["test"] = n - split_counts["train"]

        # Ajuste si la suma no da n
        if split_counts["test"] == 0:
            split_counts["train"] -= 1
            split_counts["test"] += 1


        idx = 0
        for split, count in split_counts.items():
            print(f"\nProcesando '{class_name}' - Split: {split} - Archivos: {count}")
            for _ in tqdm(range(count), desc=f"{class_name} [{split}]", unit="file"):
                bin_path = bin_files[idx]
                fname = os.path.splitext(os.path.basename(bin_path))[0]
                ann_path = os.path.join(ANNOTATION_DIR, class_name, f"annotation_{fname[-4:]}.bin")

                try:
                    events = read_ncaltech_bin(bin_path)
                    bbox = read_annotation_bbox(ann_path)
                    events_crop, crop_size = crop_events(events, bbox)

                    if len(events_crop) > 0:
                        t_max = np.uint64(np.max(events_crop[:, 2]))
                        t_min = np.uint64(np.min(events_crop[:, 2]))

                        duration = int(t_max - t_min)

                        total_duration += duration
                        valid_samples += 1


                        if duration < TIME_WINDOW:
                            continue  # omitir si el total es menor a una ventana
                        
                        
                        # clases con solo una ventana final
                        SINGLE_WINDOW_CLASSES = {
                            "airplanes", "Faces_easy", "Motorbikes",
                        }

                        if class_name in SINGLE_WINDOW_CLASSES:
                            if duration >= TIME_WINDOW:
                                t_start = np.uint64(int(t_max) - int(TIME_WINDOW))
                                start_times = np.array([t_start], dtype=np.uint64)
                            else:
                                start_times = np.array([], dtype=np.uint64)
                        else:
                            step = OFFSET_TIME_WINDOW if OFFSET_TIME_WINDOW > 0 else TIME_WINDOW
                            start_times = np.arange(t_min, t_max - TIME_WINDOW + 1, step, dtype=np.uint64)


                    if len(events_crop) < 2:
                        print(f"[WARNING] {fname}: very few events after cropping.")
                    else:
                        try:
                            for t_start in start_times:
                                t_end = t_start + TIME_WINDOW
                                window_events = events_crop[(events_crop[:, 2] >= t_start) & (events_crop[:, 2] < t_end)]

                                if len(window_events) < 2:
                                    continue  # omitir si hay muy pocos eventos

                                tag = f"{fname}_t{int(int(t_start) - int(t_min))}"
                                split_log.append(f"{split} {class_name}/{tag}.bin\n")

                                for rep in REPRESENTATIONS:
                                    rep_dir = os.path.join(OUTPUT_DIR, rep, split, class_name)
                                    os.makedirs(rep_dir, exist_ok=True)
                                    out_path = os.path.join(rep_dir, f"{tag}.png")

                                    if rep == "event_accumulate":
                                        img = make_event_accumulate(window_events, crop_size)
                                    
                                    elif rep == "sae":
                                        img = make_sae(window_events, crop_size)
                                    
                                    elif rep == "tbr":
                                        img = make_tbr(window_events, crop_size, num_bins=TBR_BINS)
                                                                        
                                    elif rep == "tbr_tensor":
                                        tensor = make_tbr_tensor(window_events, original_size=crop_size, final_size=IMG_SIZE, num_bins=TBR_BINS, rescale=True)
                                        np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                                        img = None
                                    
                                    elif rep == "tqr_tensor":
                                        tensor = make_tqr_tensor(window_events, original_size=crop_size, final_size=IMG_SIZE, num_bins=TBR_BINS, rescale=True)
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


                        except Exception as e:
                            #print(f"[ERROR] {fname} (during representation): {e}")
                            pass
                except Exception as e:
                    #print(f"[ERROR] {fname} (pre-processing): {e}")
                    pass
                finally:
                    idx += 1

    if valid_samples > 0:
        avg_duration_us = total_duration / valid_samples
        avg_duration_ms = avg_duration_us / 1000.0
        print(f"\nTiempo promedio por muestra antes de filtrar: {avg_duration_us:.0f} μs ≈ {avg_duration_ms:.2f} ms")
    else:
        print("\nNo se encontraron muestras válidas para calcular duración promedio.")


    with open(os.path.join(OUTPUT_DIR, "split_log.txt"), "w") as f:
        f.writelines(split_log)


if __name__ == "__main__":

    with open("../config_dgx-1.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="N-Caltech101 representation generator")

    parser.add_argument(
        "--input-dir",
        type=str,
        default=f"{INPUT_ROOT}",
        help="Path to the input directory (.bin files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"{OUTPUT_ROOT}",
        help="Path to the directory where to save the representations"
    )

    args = parser.parse_args()

    # === Configuración global ===
    IMG_SIZE = (240, 240) #(180, 240)
    TBR_BINS = 16
    TIME_WINDOW = np.uint64(100_000)  # en microsegundos
    OFFSET_TIME_WINDOW = np.uint64(25_000)  # en microsegundos
    INPUT_DIR = args.input_dir + "ncaltech101_bin/Caltech101"
    ANNOTATION_DIR = args.input_dir + "ncaltech101_bin/Caltech101_annotations"
    OUTPUT_DIR = args.output_dir + f"ncaltech101_rep_{str(int(TIME_WINDOW/1e3))}ms"
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]
    SPLIT_RATIO = {"train": 0.8, "test": 0.2}

    main()

