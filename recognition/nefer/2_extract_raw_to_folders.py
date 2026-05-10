import os
import glob
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

from expelliarmus import Wizard

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import make_tbr


# =========================
# CONFIG
# =========================

INPUT_RAW_ROOT = "/media/ignacio/KINGSTON/event-cameras/nefer/event_raw/raw"
OUTPUT_ROOT = "/media/ignacio/KINGSTON/event-cameras/nefer/new_dataset/event_tbr_frames"

ENCODING = "evt3"

IMG_SIZE = (720, 1280)   # (H, W) EVK4 HD
CROP_SIZE = 700

T_MS = 15 * 8            # 120 ms
TBR_BINS = 8
TIME_WINDOW_US = T_MS * 1000

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================
# UTILS
# =========================

def save_dataset_config():
    config_path = os.path.join(OUTPUT_ROOT, "dataset_config.txt")

    config_text = f"""# NEFER TBR Dataset Configuration

INPUT_RAW_ROOT = {INPUT_RAW_ROOT}
OUTPUT_ROOT = {OUTPUT_ROOT}

ENCODING = {ENCODING}

ORIGINAL_IMG_SIZE = {IMG_SIZE}
CENTER_CROP = True
CROP_SIZE = {CROP_SIZE}x{CROP_SIZE}

REPRESENTATION = tbr
TBR_BINS = {TBR_BINS}
T_MS = {T_MS}
TIME_WINDOW_US = {TIME_WINDOW_US}

WINDOWS = non-overlapping
OUTPUT_FORMAT = png

EVENT_FORMAT = [x, y, t, p]
TIMESTAMP_NORMALIZATION = t = t - t.min()
"""
    with open(config_path, "w") as f:
        f.write(config_text)


def center_crop(img, crop_size):
    h, w = img.shape[:2]

    if h < crop_size or w < crop_size:
        raise ValueError(f"Image size {img.shape} is smaller than crop_size={crop_size}")

    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2

    return img[
        start_y:start_y + crop_size,
        start_x:start_x + crop_size
    ]


def clean_img_for_pil(img):
    img = np.squeeze(img)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]

    return img.astype(np.uint8)


def read_raw_events(raw_path, encoding="evt3"):
    wiz = Wizard(encoding=encoding)
    wiz.set_file(raw_path)

    events = wiz.read()

    x = events["x"].astype(np.uint16)
    y = events["y"].astype(np.uint16)
    t = events["t"].astype(np.uint64)
    p = events["p"].astype(np.uint8)

    # Normalizar tiempo
    t = t - t.min()

    event_array = np.stack([x, y, t, p], axis=1)

    return event_array


def process_raw_file(raw_path):
    user_name = os.path.basename(os.path.dirname(raw_path))
    file_base = os.path.splitext(os.path.basename(raw_path))[0]

    out_dir = os.path.join(
        OUTPUT_ROOT,
        user_name,
        file_base
    )

    os.makedirs(out_dir, exist_ok=True)

    try:
        event_array = read_raw_events(raw_path, ENCODING)

        t_min = event_array[:, 2].min()
        t_max = event_array[:, 2].max()

        duration_us = t_max - t_min
        duration_ms = duration_us / 1000
        duration_s = duration_us / 1e6

        start_times = np.arange(
            t_min,
            t_max - TIME_WINDOW_US + 1,
            TIME_WINDOW_US,
            dtype=np.uint64
        )

        print(f"\n[FILE] {raw_path}")
        print(f"  t_min: {t_min}")
        print(f"  t_max: {t_max}")
        print(f"  duration ms: {duration_ms:.3f}")
        print(f"  duration s: {duration_s:.3f}")
        print(f"  num windows: {len(start_times)}")
        print(f"  output: {out_dir}")

        for i, t_start in enumerate(tqdm(start_times, desc=file_base, leave=False)):
            t_end = t_start + TIME_WINDOW_US

            window_events = event_array[
                (event_array[:, 2] >= t_start) &
                (event_array[:, 2] < t_end)
            ]

            if len(window_events) == 0:
                continue

            img = make_tbr(
                window_events,
                IMG_SIZE,
                TBR_BINS
            )

            img = clean_img_for_pil(img)
            img = center_crop(img, CROP_SIZE)

            out_path = os.path.join(
                out_dir,
                f"{file_base}_t{i:05d}.png"
            )

            Image.fromarray(img).save(out_path)

    except Exception as e:
        print(f"[ERROR] {raw_path}: {e}")


def main():
    save_dataset_config()

    raw_files = sorted(
        glob.glob(os.path.join(INPUT_RAW_ROOT, "user_*", "*.raw"))
    )

    print(f"Found {len(raw_files)} raw files.")

    for raw_path in tqdm(raw_files, desc="Processing RAW files"):
        process_raw_file(raw_path)

    print("DONE")


if __name__ == "__main__":
    main()