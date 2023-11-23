import os
from PIL import Image
from torchvision.transforms import v2
from torchvision import transforms
import numpy as np
import random


def annotate_img(
    img_filename, center=None, trap_size=None, img_size=(224, 224), class_name="0"
):
    """
    Creates an annotation file, or append the annotation if file already exists, from the given bb info
    """
    path = img_filename.split(".")[0]
    f = open(path + ".txt", "a")

    if center and trap_size:
        dw = 1.0 / img_size[0]
        dh = 1.0 / img_size[1]
        x = center[0] * dw
        y = center[1] * dh
        w = trap_size[0] * dw
        h = trap_size[1] * dh

        annotation = " ".join([class_name, str(x), str(y), str(h), str(w), "\n"])
        f.write(annotation)

    f.close()


def get_random_coordinates(img_size, back_size):
    """
    Return random coordinates (w, h, left, top, center)
    """
    w, h = img_size
    left = random.randint(0, back_size[0] - w)
    top = random.randint(0, back_size[1] - h)
    center = (left + int(w / 2), top + int(h / 2))

    return w, h, left, top, center


def get_random_sample(samples_dir):
    """
    Return a random img form the images in a given directory
    """

    samples = os.listdir(samples_dir)

    sample_name = random.choice(samples)
    img = Image.open(os.path.join(samples_dir, sample_name))

    img_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            v2.RandomResize(15, 30, interpolation=Image.BICUBIC, antialias=True),
        ]
    )
    img = img_transforms(img)

    angle = random.randint(0, 360)
    img.rotate(angle, resample=Image.BICUBIC, expand=True)

    return img


def add_items(back, item_dir, n_items, ann_path=None):
    """
    Places n random items on a given image
    """
    for i in range(n_items):
        item = get_random_sample(item_dir)
        w, h, left, top, center = get_random_coordinates(item.size, back.size)

        if ann_path:
            annotate_img(ann_path, center, (w, h), back.size, "0")

        # Combine and save background + bush
        back.paste(item, (left, top), item)

    return back


def load_background(back_dir, sample, transform_p=0.75):
    img_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(transform_p),
            transforms.RandomVerticalFlip(transform_p),
        ]
    )

    back = Image.open(os.path.join(back_dir, sample))

    return img_transforms(back)


def augment_imgs(imgs_dir, save_img_dir, policy=v2.AutoAugmentPolicy.IMAGENET):
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    augment = v2.AutoAugment(policy)

    samples_imgs = os.listdir(imgs_dir)
    print("N samples: ", len(samples_imgs))

    for sample in samples_imgs:
        # Load and transform background, trap and bush images
        img = Image.open(os.path.join(imgs_dir, sample))

        img = augment(img)
        new_img_name = sample.split(".")[0] + "_" + "aug.png"
        img.save(os.path.join(save_img_dir, new_img_name))


def add_bushes(
    back_dir,
    bush_dir,
    save_img_dir,
    n_min=1,
    n_max=5,
    p_distribution=[0.4, 0.3, 0.2, 0.1],
    transform_p=0.75,
):
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    jitter_range = (0.8, 1.2)
    color_jitter = transforms.ColorJitter(
        brightness=jitter_range,
        contrast=jitter_range,
        saturation=jitter_range,
        hue=(-0.5, 0.5),
    )

    samples_back = os.listdir(back_dir)
    print("N samples: ", len(samples_back))

    for sample in samples_back:
        back = load_background(back_dir, sample, transform_p)

        n_items = np.random.choice(np.arange(n_min, n_max), p=p_distribution)
        back = add_items(
            back,
            bush_dir,
            n_items,
        )

        back = color_jitter(back)

        new_name = (
            sample.split(".")[0] + "_" + str(n_items) + "b.png"
        )  # ie 12345_2b.png
        print(new_name)

        back.save(os.path.join(save_img_dir, new_name))


def add_traps(
    back_dir,
    trap_dir,
    save_img_dir,
    save_ann_dir=None,
    n_min=1,
    n_max=4,
    p_distribution=[0.75, 0.15, 0.1],
    transform_p=0.75,
):
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)

    if save_ann_dir and not os.path.exists(save_ann_dir):
        os.makedirs(save_ann_dir)

    jitter_range = (0.75, 1.25)
    color_jitter = transforms.ColorJitter(
        brightness=jitter_range,
        contrast=jitter_range,
        saturation=jitter_range,
        hue=(-0.5, 0.5),
    )

    samples_back = os.listdir(back_dir)
    print("N samples: ", len(samples_back))

    for sample in samples_back:
        # Load and transform background, trap and bush images
        back = load_background(back_dir, sample, transform_p)

        n_items = np.random.choice(np.arange(n_min, n_max), p=p_distribution)

        new_name = (
            sample.split(".")[0] + "_" + str(n_items) + "t.png"
        )  # SampleID_2t.png
        print(new_name)

        if save_ann_dir:
            back = add_items(
                back, trap_dir, n_items, os.path.join(save_ann_dir, new_name)
            )
        else:
            back = add_items(back, trap_dir, n_items)

        back = color_jitter(back)

        back.save(os.path.join(save_img_dir, new_name))
