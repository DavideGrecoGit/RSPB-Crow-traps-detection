from patchify import patchify
import argparse
import os
import tifffile as tiff
from PIL import Image
from math import floor


def make_patches(dataset_dir, save_dir, patch_size=224, img_format="jpg"):
    """
    Generate patches of the given size from the images into the given directory
    """

    os.makedirs(save_dir, exist_ok=True)

    original_sample_names = os.listdir(dataset_dir)

    for name in original_sample_names:
        sample_path = os.path.join(dataset_dir, name)

        if os.path.isdir(sample_path):
            continue

        image = tiff.imread(sample_path, key=0)

        h, w, c = image.shape

        if h >= patch_size and w >= patch_size:
            h_new = floor(h / patch_size) * patch_size
            w_new = floor(w / patch_size) * patch_size
            image = image[:h_new, :w_new, :]

            patches_imgs = patchify(image, (224, 224, 3), step=224)
            print("N patches ", patches_imgs.shape)

            for i in range(len(patches_imgs)):
                for j in range(len(patches_imgs[i])):
                    im = Image.fromarray(patches_imgs[i][j][0])
                    save_path = os.path.join(
                        save_dir, f"{name.split('.')[0]}_{i}_{j}.{img_format}"
                    )
                    im.save(save_path)
        else:
            print("Skipping image: patch size too big!")

    return os.listdir(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", help="Path to images to patchify", type=str, required=True
    )

    parser.add_argument("-o", help="Path to output directory", type=str, default=None)

    parser.add_argument("-patch_size", help="Patch size", type=int, default=224)

    args = parser.parse_args()

    if args.o is None:
        args.o = os.path.join(args.i, "patches")

    patches = make_patches(args.i, args.o, patch_size=args.patch_size)
    print("Total patches: ", len(patches))
