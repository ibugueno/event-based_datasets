import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import zoom
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
OUTPUT_DIR = {OUTPUT_DIR}
IMG_SIZE (final) = {IMG_SIZE}
TBR_BINS = {TBR_BINS}
TIME_WINDOW = {int(TIME_WINDOW)} (μs)
OFFSET_TIME_WINDOW = {int(OFFSET_TIME_WINDOW)} (μs)
REPRESENTATIONS = {', '.join(REPRESENTATIONS)}
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

def crop_and_resize_to_resolution(img, output_size=(224, 224)):
    """
    Recorta cuadrado centrado (sin padding) y luego resize, igual que el método crop_and_resize_to_resolution del modelo.
    """
    img = np.squeeze(img)

    # Asegurar formato (H, W, C)
    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[2] == 1:  # (H, W, 1) → (H, W)
        img = img[:, :, 0]

    h, w = img.shape[:2]

    # Recorte cuadrado centrado (fiel al modelo)
    if h > w:
        center = h // 2
        half = w // 2
        img = img[center - half:center + half, :]
    elif w > h:
        center = w // 2
        half = h // 2
        img = img[:, center - half:center + half]

    # Resize final
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize(output_size, Image.Resampling.BILINEAR)
    return np.array(img)




def crop_and_resize_tensor(tensor, output_size=(224, 224)):
    """
    Aplica crop cuadrado centrado y resize usando interpolación 'nearest',
    con lógica idéntica a crop_and_resize_to_resolution (modelo).
    Soporta tensores (H, W, C) o (H, W, C, 2).
    """
    h, w = tensor.shape[:2]

    # --- Recorte cuadrado centrado fiel al modelo ---
    if h > w:
        center = h // 2
        half = w // 2
        tensor = tensor[center - half:center + half, :, ...]
    elif w > h:
        center = w // 2
        half = h // 2
        tensor = tensor[:, center - half:center + half, ...]

    h_cropped, w_cropped = tensor.shape[:2]
    zoom_factors = [output_size[0] / h_cropped, output_size[1] / w_cropped]

    # --- Expandir dimensiones según corresponda ---
    if tensor.ndim == 3:
        zoom_factors += [1]
    elif tensor.ndim == 4:
        zoom_factors += [1, 1]
    else:
        raise ValueError(f"Tensor con forma no soportada: {tensor.shape}")

    tensor_resized = zoom(tensor, zoom=zoom_factors, order=0)  # Interpolación nearest
    return tensor_resized.astype(tensor.dtype)



# === Main ===
def main():
    save_dataset_config()  

    split_log = []
    total_duration = 0
    valid_samples = 0

    for split in ["training", "validation", "testing"]:
        split_dir = os.path.join(INPUT_DIR, split)
        print(f"\nProcesando split: {split}")

        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            npy_files = sorted(glob.glob(os.path.join(class_dir, "*.npy")))

            print(f"\n  Clase: {class_name} - Archivos: {len(npy_files)}")
            for npy_path in tqdm(npy_files, desc=f"{class_name} [{split}]", unit="file"):
                fname = os.path.splitext(os.path.basename(npy_path))[0]

                try:

                    raw_events = np.load(npy_path)

                    try:
                        x = raw_events[:, 0].astype(np.uint16)  # x
                        y = raw_events[:, 1].astype(np.uint16)  # y
                        t = (raw_events[:, 2] * 1e6).astype(np.uint64)  # t: segundos → microsegundos
                        p = (raw_events[:, 3]).astype(np.bool_)  # t: segundos → microsegundos
                        events = np.stack([x, y, t, p], axis=-1)
                    except Exception as e:
                        print(f"[ERROR] {fname}: error al procesar eventos - {e}")
                        continue


                    # Calcular duración y validar
                    t_min = np.min(events[:, 2])
                    t_max = np.max(events[:, 2])
                    duration = t_max - t_min

                    if duration < TIME_WINDOW:
                        continue


                    total_duration += duration
                    valid_samples += 1

                    step = OFFSET_TIME_WINDOW if OFFSET_TIME_WINDOW > 0 else TIME_WINDOW
                    start_times = np.arange(t_min, t_max - TIME_WINDOW, step)

                    for t_start in start_times:
                        t_end = t_start + TIME_WINDOW
                        window_events = events[(events[:, 2] >= t_start) & (events[:, 2] < t_end)]


                        if len(window_events) < 2:
                            continue

                        tag = f"{fname}_t{int(t_start - t_min)}"
                        split_log.append(f"{split} {class_name}/{tag}.npy\n")

                        for rep in REPRESENTATIONS:
                            rep_dir = os.path.join(OUTPUT_DIR, rep, split, class_name)
                            os.makedirs(rep_dir, exist_ok=True)
                            out_path = os.path.join(rep_dir, f"{tag}.png")

                            if rep == "event_accumulate":
                                img = make_event_accumulate(window_events, img_size=(180, 240))

                            elif rep == "sae":
                                img = make_sae(window_events, img_size=(180, 240))

                            elif rep == "tbr":
                                img = make_tbr(window_events, img_size=(180, 240), num_bins=TBR_BINS)

                            elif rep == "tbr_tensor":
                                tensor = make_tbr_tensor(
                                    window_events,
                                    original_size=(180, 240),
                                    final_size=(180, 240),  
                                    num_bins=TBR_BINS,
                                    rescale=False
                                )
                                tensor = crop_and_resize_tensor(tensor, IMG_SIZE)
                                np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                                img = None

                            elif rep == "tqr_tensor":
                                tensor = make_tqr_tensor(
                                    window_events,
                                    original_size=(180, 240),
                                    final_size=(180, 240),
                                    num_bins=TBR_BINS,
                                    rescale=False
                                )
                                tensor = crop_and_resize_tensor(tensor, IMG_SIZE)
                                np.save(os.path.join(rep_dir, f"{tag}.npy"), tensor)
                                img = None


                            elif rep == "tencode":
                                img = make_tencode(window_events, img_size=(180, 240))

                            elif rep == "behi":
                                img = make_behi(window_events, img_size=(180, 240))

                            else:
                                img = None

                            #print(rep)
                            if img is not None:
                                #print(out_path)
                                resized = crop_and_resize_to_resolution(img, IMG_SIZE)
                                if resized.ndim == 3:
                                    save_color_image(resized, out_path)
                                else:
                                    save_image(resized, out_path)

                except Exception as e:
                    print(f"[ERROR] {fname}: {e}")
                    continue

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
    IMG_SIZE = (224, 224) #(180, 240)
    TBR_BINS = 16
    TIME_WINDOW = np.uint64(100_000)  # en microsegundos
    OFFSET_TIME_WINDOW = np.uint64(0)#np.uint64(25_000)  # en microsegundos
    INPUT_DIR = args.input_dir + "N-Caltech101"
    OUTPUT_DIR = args.output_dir + f"ncaltech101npy_rep_{str(int(TIME_WINDOW/1e3))}ms"
    REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]

    main()

