import os
import numpy as np
import random
from tqdm import tqdm
from PIL import Image
import glob
import sys
import argparse
import yaml
import traceback

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

IMG_SIZE = {IMG_SIZE}

TBR_BINS = {TBR_BINS}

TIME_WINDOW = {int(TIME_WINDOW)} (μs)

REPRESENTATIONS = {', '.join(REPRESENTATIONS)}

CLASSES = {CLASS_NAMES}
"""

    with open(path, "w") as f:
        f.write(config_text)


def process_file(file_path, split, label):

    try:

        events = np.load(file_path, allow_pickle=True)

        # REDAT format:
        # (4, N) -> [x, y, t, p]

        if events.shape[0] == 4:

            x = events[0].astype(np.uint16)
            y = events[1].astype(np.uint16)
            t = events[2].astype(np.uint64)
            p = events[3].astype(np.uint8)

        elif events.shape[1] == 4:

            x = events[:, 0].astype(np.uint16)
            y = events[:, 1].astype(np.uint16)
            t = events[:, 2].astype(np.uint64)
            p = events[:, 3].astype(np.uint8)

        else:

            raise ValueError(f"Unknown event format: {events.shape}")

        # Orientation correction
        H, W = IMG_SIZE

        # Vertical flip
        y = (H - 1) - y

        # Horizontal flip
        x = (W - 1) - x

        # sanity checks

        assert x.max() < 240
        assert y.max() < 180

        assert np.all(np.diff(t) >= 0)

        assert set(np.unique(p)).issubset({0, 1})

        event_array = np.stack([x, y, t, p], axis=1)

        t_min = event_array[:, 2].min()
        t_max = event_array[:, 2].max()

        start_times = np.arange(
            t_min,
            t_max - TIME_WINDOW + 1,
            TIME_WINDOW,
            dtype=np.uint64
        )

        parent_folder = os.path.basename(os.path.dirname(file_path))
        file_base = os.path.splitext(os.path.basename(file_path))[0]

        for i, t_start in enumerate(start_times):

            t_end = t_start + TIME_WINDOW

            window_events = event_array[
                (event_array[:, 2] >= t_start) &
                (event_array[:, 2] < t_end)
            ]

            if len(window_events) == 0:
                continue

            tag = f"{parent_folder}_{file_base}_t{i:03d}"

            for rep in REPRESENTATIONS:

                rep_dir = os.path.join(
                    OUTPUT_DIR,
                    rep,
                    split,
                    str(label)
                )

                os.makedirs(rep_dir, exist_ok=True)

                out_path = os.path.join(rep_dir, f"{tag}.png")

                if rep == "event_accumulate":

                    img = make_event_accumulate(
                        window_events,
                        IMG_SIZE
                    )

                    save_image(img, out_path)

                elif rep == "sae":

                    img = make_sae(
                        window_events,
                        IMG_SIZE
                    )

                    save_image(img, out_path)

                elif rep == "tbr":

                    img = make_tbr(
                        window_events,
                        IMG_SIZE,
                        TBR_BINS
                    )

                    save_image(img, out_path)

                elif rep == "tbr_tensor":

                    tensor = make_tbr_tensor(
                        window_events,
                        IMG_SIZE,
                        IMG_SIZE,
                        TBR_BINS,
                        rescale=False
                    )

                    np.save(
                        os.path.join(rep_dir, f"{tag}.npy"),
                        tensor
                    )

                elif rep == "tqr_tensor":

                    tensor = make_tqr_tensor(
                        window_events,
                        IMG_SIZE,
                        IMG_SIZE,
                        TBR_BINS,
                        rescale=False
                    )

                    np.save(
                        os.path.join(rep_dir, f"{tag}.npy"),
                        tensor
                    )

                elif rep == "tencode":

                    img = make_tencode(
                        window_events,
                        IMG_SIZE
                    )

                    save_color_image(img, out_path)

                elif rep == "behi":

                    img = make_behi(
                        window_events,
                        IMG_SIZE
                    )

                    save_image(img, out_path)

    except Exception as e:
        print(f"\n[ERROR] {file_path}: {e}")
        traceback.print_exc()


def main():

    save_dataset_config()

    class_mapping = {
        "idle": 0,
        "pick": 1,
        "place": 2,
        "screw": 3
    }

    for split in ["train", "test"]:

        for class_name in CLASS_NAMES:

            class_dir = os.path.join(INPUT_DIR, split, class_name)

            files = sorted(
                glob.glob(os.path.join(class_dir, "*_events.npy"))
            )

            label = class_mapping[class_name]

            print(f"\n[{split}] {class_name}: {len(files)} files")

            for file_path in tqdm(files, desc=f"{split}-{class_name}"):

                process_file(
                    file_path=file_path,
                    split=split,
                    label=label
                )


if __name__ == "__main__":

    with open("../config_dgx-1.yaml", "r") as f:
        config = yaml.safe_load(f)

    INPUT_ROOT = config["paths"]["input_root"]
    OUTPUT_ROOT = config["paths"]["output_root"]

    parser = argparse.ArgumentParser(description="REDAT24 representation generator")
    parser.add_argument("--input-dir", type=str, default=f"{INPUT_ROOT}", help="Path to input dir")
    parser.add_argument("--output-dir", type=str, default=f"{OUTPUT_ROOT}", help="Path to output dir")
    parser.add_argument("--timewindow-ms", type=int, default=100, help="Window size in ms (default: 100)")

    args = parser.parse_args()

    width, height = 240, 180

    IMG_SIZE = (height, width)
    TBR_BINS = 8
    TIME_WINDOW = np.uint64(args.timewindow_ms * 1_000)
    INPUT_DIR = os.path.join(args.input_dir, "redat24_npy_split")
    OUTPUT_DIR = os.path.join(args.output_dir, f"redat24_rep_{int(TIME_WINDOW/1e3)}ms")

    CLASS_NAMES = [
        "idle",
        "pick",
        "place",
        "screw"
    ]

    print(f'INPUT_DIR: {INPUT_DIR}')
    print(f'OUTPUT_DIR: {OUTPUT_DIR}')

    #REPRESENTATIONS = ["event_accumulate", "sae", "tbr", "tbr_tensor", "tqr_tensor", "tencode", "behi"]
    #REPRESENTATIONS = ["event_accumulate", "tbr", "tbr_tensor", "tqr_tensor", "tencode"]
    REPRESENTATIONS = ["tqr_tensor"]

    main()