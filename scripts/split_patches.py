from sklearn.model_selection import train_test_split
import os
import json
import argparse
import shutil
from tqdm import tqdm

SEED = 42


def save_split_ids(ids_split, save_path):
    with open(save_path, "w") as f:
        json.dump(ids_split, f, indent=2)


def load_split_ids(load_path):
    with open(load_path, "r") as f:
        ids = json.load(f)
    return ids


def save_split_images(list_ids, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    samples_imgs = os.listdir(input_dir)

    for sample in samples_imgs:
        sample_id = sample.split("_")[0]
        if sample_id in list_ids:
            shutil.copy(os.path.join(input_dir, sample), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-data_dir",
        help="Path to aerial images directory. Filenames should will be threated as IDs",
        required=True,
    )
    parser.add_argument(
        "-split_size",
        help="Test split size, also used to as validation split size.",
        type=int,
        default=15,
    )
    parser.add_argument("-output_dir", help="Path to output directory.", required=True)
    parser.add_argument(
        "-patches_dir", help="Path to patches directory.", required=True
    )

    parser.add_argument(
        "-add_only_traps",
        help="Additionally save 'only_traps' data",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    # Get list of ids
    files = [file.split(".")[0] for file in os.listdir(args.data_dir)]

    # Split ids into train, val and test
    splits = {}
    splits["train"], ids_temp = train_test_split(
        files, test_size=args.split_size * 2, random_state=SEED
    )
    splits["val"], splits["test"] = train_test_split(
        ids_temp, test_size=0.5, random_state=SEED
    )

    # Save ids
    ids_path = os.path.join(args.output_dir, "ids_sets")
    os.makedirs(ids_path, exist_ok=True)

    for split_type in splits.keys():
        save_split_ids(
            splits[split_type], os.path.join(ids_path, f"ids_{split_type}.txt")
        )

    img_types = ["background", "crow_trap"]
    if args.add_only_traps:
        img_types.append("only_traps")

    for i in tqdm(range(len(img_types))):
        for split_type in splits.keys():
            save_split_images(
                splits[split_type],
                os.path.join(args.patches_dir, img_types[i]),
                os.path.join(args.output_dir, split_type, img_types[i]),
            )
