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

def save_dataset_config():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = f"{OUTPUT_DIR}/dataset_config.txt"
    config_text = f"""# Dataset Parameters
INPUT_DIR = {INPUT_DIR}
OUTPUT_DIR = {OUTPUT_DIR}
IMG_SIZE (final) = {IMG_SIZE}
TBR_BINS = {TBR_BINS}
TIME_WINDOW = {int(TIME_WINDOW)} (\u03bcs)
REPRESENTATIONS = {', '.join(REPRESENTATIONS)}
    """
    with open(path, "w") as f:
        f.write(config_text)

def parse_aedat_events(filepath, retina_size_x=128):
    try:
        with open(filepath, 'rb') as f:
            bof = f.tell()
            line = f.readline()
            # Saltar cabecera
            while line and line.startswith(b"#"):
                bof = f.tell()
                line = f.readline()

            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(bof, os.SEEK_SET)

            num_events = (file_size - bof) // 8
            data = np.fromfile(f, dtype='>u4', count=2*num_events)

        if data.size < 2:
            print(f"[WARNING] Archivo vacío o muy corto: {filepath}")
            return np.empty((0, 4), dtype=np.uint64)

        all_addr = data[::2]
        all_ts = data[1::2]

        # Máscara y shift para decodificar eventos
        xmask = 0xFE
        ymask = 0x7F00
        xshift = 1
        yshift = 8
        polmask = 0x1

        if retina_size_x == 32:
            xshift = 3
            yshift = 10

        addr = np.abs(all_addr)
        x = ((addr & xmask) >> xshift).astype(np.uint16)
        y = ((addr & ymask) >> yshift).astype(np.uint16)
        p = ((addr & polmask) == 0).astype(np.uint8)  # 1: ON, 0: OFF

        # timestamps relativos
        t = (all_ts - all_ts[0]).astype(np.uint64)

        # Filtrar eventos que estén dentro del rango de la imagen
        valid_mask = (x < IMG_SIZE[1]) & (y < IMG_SIZE[0])
        x = x[valid_mask]
        y = y[valid_mask]
        t = t[valid_mask]
        p = p[valid_mask]

        events_array = np.stack([x, y, t, p], axis=1)
        return events_array


    except Exception as e:
        print(f"[ERROR] Error leyendo {filepath} como binario 32b: {e}")
        return np.empty((0, 4), dtype=np.uint64)




def save_image(array, path):
    array = np.squeeze(array)
    array = np.rot90(array, k=1)
    Image.fromarray(array.astype(np.uint8)).save(path)

def save_color_image(array, path):
    array = np.rot90(array, k=1)
    Image.fromarray(array).save(path)

def get_label_from_filename(filename):
    filename = filename.lower()
    if 'club' in filename:
        return 'club'
    elif 'diamond' in filename:
        return 'diamond'
    elif 'heart' in filename:
        return 'heart'
    elif 'spade' in filename:
        return 'spade'
    else:
        return 'unknown'

def main():
    save_dataset_config()

    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.aedat')))
    class_files = {'club': [], 'diamond': [], 'heart': [], 'spade': []}

    for f in all_files:
        label = get_label_from_filename(os.path.basename(f))
        if label in class_files:
            class_files[label].append(f)

    split_log = []

    for label, files in class_files.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * SPLIT_RATIO['train'])
        n_val = n - n_train  # sin test
        split_data = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
        }

        for split, file_list in split_data.items():
            for path in tqdm(file_list, desc=f"{label} - {split}"):
                fname = os.path.splitext(os.path.basename(path))[0]
                try:
                    if os.path.getsize(path) < 1024:
                        print(f"[WARNING] Archivo muy pequeño, posiblemente corrupto: {fname}")
                        continue

                    events = parse_aedat_events(path)

                    if events.shape[0] == 0 or events.shape[1] != 4:
                        print(f"[WARNING] Formato inválido o evento vacío en: {fname}")
                        continue

                    # Tomar solo eventos dentro de los primeros 100ms
                    t0 = np.uint64(events[:, 2].min())
                    t_max = t0 + TIME_WINDOW
                    events = events[events[:, 2] <= t_max]

                    if len(events) == 0:
                        continue

                    for rep in REPRESENTATIONS:
                        rep_dir = os.path.join(OUTPUT_DIR, rep, split, label)
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
                            tensor = make_tbr_tensor(events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                            # np.save(os.path.join(rep_dir, f"{fname}.npy"), tensor)
                        elif rep == "tqr_tensor":
                            tensor = make_tqr_tensor(events, IMG_SIZE, IMG_SIZE, TBR_BINS, rescale=False)
                            # np.save(os.path.join(rep_dir, f"{fname}.npy"), tensor)
                        elif rep == "tencode":
                            img = make_tencode(events, IMG_SIZE)
                            save_color_image(img, out_path)
                        elif rep == "behi":
                            img = make_behi(events, IMG_SIZE)
                            save_image(img, out_path)

                    split_log.append(f"{split} {label}/{fname}.aedat\n")
                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")

    with open(os.path.join(OUTPUT_DIR, "split_log.txt"), "w") as f:
        f.writelines(split_log)



if __name__ == "__main__":

    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="PokerDVS representation generator")
    parser.add_argument("--input-dir", type=str, default=f"{INPUT_ROOT}/pokerdvs/", help="Path to input dir")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_ROOT}/pokerdvs/", help="Path to output dir")
    args = parser.parse_args()

    IMG_SIZE = (32, 32)
    TBR_BINS = 8
    TIME_WINDOW = np.uint64(100_000)  # 100 ms
    INPUT_DIR = os.path.join(args.input_dir, "pokerdvs_aedat")
    OUTPUT_DIR = os.path.join(args.output_dir, f"pokerdvs_rep_{int(TIME_WINDOW/1e3)}ms")
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]
    SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}

    main()
