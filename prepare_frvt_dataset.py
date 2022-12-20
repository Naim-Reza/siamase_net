import os
from shutil import copy2
from tqdm import tqdm

walk_dir = '/home/naim/cvlab/frvt/common/images/'
dest_dir = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/frvt_classes/'
exclude_dir = 'video1'
all_images = {}

for roots, subdirs, files in os.walk(walk_dir):
    if exclude_dir in subdirs:
        subdirs.remove(exclude_dir)

    for file in files:
        if not file.find('S') == -1:
            class_name = file.split('-')[0]
            path = os.path.join(roots, file)
            if class_name not in all_images.keys():
                all_images[class_name] = []
            all_images[class_name].append(path)

for className, images in tqdm(all_images.items()):
    classDir = os.path.join(dest_dir, className)
    if not os.path.exists(classDir):
        os.mkdir(classDir)

    for src in images:
        img_name = src.split('/')[-1]
        dest_path = os.path.join(classDir, img_name)
        copy2(src, dest_path)
