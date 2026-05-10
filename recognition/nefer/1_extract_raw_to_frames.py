import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys

from expelliarmus import Wizard

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import make_tbr

# =========================
# CONFIG
# =========================

raw_path = "/media/ignacio/KINGSTON/event-cameras/nefer/event_raw/raw/user_21/user21_2022-06-17_12-13-03.raw"

output_dir = "./tbr_20ms_8bins"

IMG_SIZE = (720, 1280)   # (H, W) EVK4
T_MS = 15*8
TBR_BINS = 8

TIME_WINDOW_US = T_MS * 1000

os.makedirs(output_dir, exist_ok=True)

# =========================
# LOAD RAW
# =========================

wiz = Wizard(encoding="evt3")
wiz.set_file(raw_path)

events = wiz.read()

# =========================
# EXTRACT EVENTS
# =========================

x = events["x"].astype(np.uint16)
y = events["y"].astype(np.uint16)
t = events["t"].astype(np.uint64)
p = events["p"].astype(np.uint8)

# normalizar tiempo
t = t - t.min()

event_array = np.stack([x, y, t, p], axis=1)

t_min = event_array[:, 2].min()
t_max = event_array[:, 2].max()

print("t_min:", t_min)
print("t_max:", t_max)

print("duración us:", t_max - t_min)
print("duración ms:", (t_max - t_min) / 1000)
print("duración s:", (t_max - t_min) / 1e6)

# =========================
# NON-OVERLAPPING WINDOWS
# =========================

start_times = np.arange(
    t_min,
    t_max - TIME_WINDOW_US + 1,
    TIME_WINDOW_US,
    dtype=np.uint64
)

print("num windows:", len(start_times))

# =========================
# GENERATE TBR PNG
# =========================

base_name = os.path.splitext(os.path.basename(raw_path))[0]

for i, t_start in enumerate(tqdm(start_times)):

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

    out_path = os.path.join(
        output_dir,
        f"{base_name}_t{i:05d}.png"
    )

    img = np.squeeze(img)

    if img.ndim == 3 and img.shape[-1] == 1:
        img = img[:, :, 0]

    # =========================
    # CENTER CROP 700x700
    # =========================

    crop_size = 700

    h, w = img.shape[:2]

    start_y = (h - crop_size) // 2
    start_x = (w - crop_size) // 2

    img = img[
        start_y:start_y + crop_size,
        start_x:start_x + crop_size
    ]

    Image.fromarray(img.astype(np.uint8)).save(out_path)

print("DONE")