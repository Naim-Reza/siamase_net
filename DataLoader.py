import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SiameseDataset(Dataset):
    def __init__(self, data_root, image_size, train=True, test=False):
        self.data_root = data_root
        self.train = train
        self.test = test
        self.transforms = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.image_tuples = generate_tuples(data_root)
        self.train_tuples = self.image_tuples[0: round(len(self.image_tuples) * 0.8)]
        self.val_tuples = self.image_tuples[round(len(self.image_tuples) * 0.8):]

    def __len__(self):
        if not self.test:
            return len(self.train_tuples) if self.train else len(self.val_tuples)

        return len(self.image_tuples)

    def __getitem__(self, index):
        if not self.test:
            anchore_image_path, positive_image_path, negative_image_path = self.train_tuples[index] if self.train else \
                self.val_tuples[index]
        else:
            anchore_image_path, positive_image_path, negative_image_path = self.image_tuples[index]

        anchore_image = self.preprocess_image(anchore_image_path)
        positive_image = self.preprocess_image(positive_image_path)
        negative_image = self.preprocess_image(negative_image_path)
        positive_class_name = anchore_image_path.split('/')[-2]
        negative_class_name = negative_image_path.split('/')[-2]

        # if not self.test:
        #     return anchore_image, positive_image, negative_image
        return anchore_image, positive_image, negative_image, int(positive_class_name), int(negative_class_name)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        transformed = self.transforms(image)
        return transformed


def generate_tuples(root_dir):
    sub = list()
    all_files = list()
    for roots, subdirs, files in os.walk(root_dir):
        subdirs.sort()
        sub.append(subdirs)
        root = list()
        for file in files:
            root.append(os.path.join(roots, file))

        all_files.append(root)

    classes = sub[0]

    a_images = list()
    p_images = list()

    if not all_files[0]: all_files.pop(0)

    for i, sl in enumerate(classes):
        f = all_files[i]
        if f:
            subset_len = len(f) // 2
            a_images.append(f[0: subset_len])
            p_images.append(f[subset_len: subset_len + subset_len])

    a_images = flatten(a_images)
    p_images = flatten(p_images)
    n_images = p_images + a_images
    n_images = shuffle(n_images, 42)
    n_images = n_images[0: len(a_images)]
    n_images = shuffle(n_images, 64)
    # img_tuple = zip(a_images, p_images, n_images)
    img_tuple = zip(reversed(a_images), reversed(p_images), reversed(n_images))
    return list(img_tuple)


def flatten(l):
    return [item for sublist in l for item in sublist]


def shuffle(arr, seed):
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(arr)
    return arr
