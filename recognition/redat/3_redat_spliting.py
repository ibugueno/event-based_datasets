# Example:
#
# python 3_redat_spliting.py \
#     --input-dir /media/ignacio/KINGSTON/event-cameras/redat/npy \
#     --output-dir /media/ignacio/KINGSTON/event-cameras/redat/npy_split \
#     --train-ratio 0.8

import os
import glob
import shutil
import argparse
from tqdm import tqdm

def make_dir(path):
    os.makedirs(path, exist_ok=True)


def split_class(class_dir, output_dir, train_ratio=0.8, copy=True):

    class_name = os.path.basename(class_dir)

    files = sorted(glob.glob(os.path.join(class_dir, "*")))

    # Agrupar archivos por muestra
    # Ejemplo:
    # idle_1_events.npy
    # idle_1_labels.csv

    samples = {}

    for f in files:

        base = os.path.basename(f)

        if base.endswith("_events.npy"):

            key = base.replace("_events.npy", "")

            samples.setdefault(key, {})["events"] = f

        elif base.endswith("_labels.csv"):

            key = base.replace("_labels.csv", "")

            samples.setdefault(key, {})["labels"] = f

    sample_keys = sorted(samples.keys())

    n_train = int(len(sample_keys) * train_ratio)

    train_keys = sample_keys[:n_train]
    test_keys = sample_keys[n_train:]

    split_dict = {
        "train": train_keys,
        "test": test_keys
    }

    print(
        f"{class_name}: "
        f"total={len(sample_keys)} | "
        f"train={len(train_keys)} | "
        f"test={len(test_keys)}"
    )

    for split, keys in split_dict.items():

        split_class_dir = os.path.join(
            output_dir,
            split,
            class_name
        )

        make_dir(split_class_dir)

        for key in tqdm(keys, desc=f"{split}-{class_name}"):

            for file_type in ["events", "labels"]:

                if file_type not in samples[key]:

                    print(f"[WARNING] Missing {file_type}: {class_name}/{key}")

                    continue

                src = samples[key][file_type]

                dst = os.path.join(
                    split_class_dir,
                    os.path.basename(src)
                )

                if copy:
                    shutil.copy2(src, dst)
                else:
                    shutil.move(src, dst)


def main():

    parser = argparse.ArgumentParser(
        description="REDAT24 train/test splitter"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="/media/ignacio/KINGSTON/event-cameras/redat/npy",
        help="Input folder with class subfolders"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/ignacio/KINGSTON/event-cameras/redat/npy_split",
        help="Output folder"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio"
    )

    parser.add_argument(
        "--move",
        action="store_true",
        help="Move instead of copy"
    )

    args = parser.parse_args()

    class_names = [
        "idle",
        "pick",
        "place",
        "screw"
    ]

    make_dir(args.output_dir)

    for class_name in class_names:

        class_dir = os.path.join(
            args.input_dir,
            class_name
        )

        if not os.path.isdir(class_dir):

            print(f"[WARNING] Missing class folder: {class_dir}")

            continue

        split_class(
            class_dir=class_dir,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            copy=not args.move
        )


if __name__ == "__main__":
    main()