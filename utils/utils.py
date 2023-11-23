from patchify import patchify
import tifffile as tiff
import os
from PIL import Image, ImageDraw
from math import floor
import shutil
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import random
from torchvision import transforms


def make_split(
    class_name, dataset_dir="Datasets/Crow_classify", classify=False, split_p=0.2
):
    """
    Saves the data in a given path into train and test folders
    """

    class_dir = os.path.join(dataset_dir, class_name)
    samples_list = os.listdir(class_dir)

    print("Sample list: ", len(samples_list))

    samples_train, samples_test = train_test_split(
        samples_list, test_size=split_p, random_state=42
    )

    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    if classify:
        train_dir = os.path.join(dataset_dir, "train", class_name)
        test_dir = os.path.join(dataset_dir, "test", class_name)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    print(len(samples_train), " in ", train_dir)
    print(len(samples_test), " in ", test_dir)

    for sample in samples_test:
        sample_path = os.path.join(class_dir, sample)
        save_path = os.path.join(test_dir, sample)

        shutil.copyfile(sample_path, save_path)

    for sample in samples_train:
        sample_path = os.path.join(class_dir, sample)
        save_path = os.path.join(train_dir, sample)

        shutil.copyfile(sample_path, save_path)


def make_patches(dataset_dir, save_dir, patch_size=224, img_format="jpeg"):
    """
    Generate patches of the given size from the images into the given directory
    """

    if not os.path.exists(save_dir):
        print("Creating save directory")
        os.makedirs(save_dir)

    original_sample_names = os.listdir(dataset_dir)

    for name in original_sample_names:
        sample_path = os.path.join(dataset_dir, name)
        image = tiff.imread(sample_path, key=0)
        print(image.shape)
        h, w, c = image.shape

        if h >= patch_size and w >= patch_size:
            h_new = floor(h / patch_size) * patch_size
            w_new = floor(w / patch_size) * patch_size
            image = image[:h_new, :w_new, :]
            print(image.shape)

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
