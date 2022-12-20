from PIL import Image
from torchvision import transforms
import torch
import numpy as np

from model import resnet, head_models, siamasenet

import os

if __name__ == '__main__':
    pos_dir = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/setA_less/98/'
    neg_dir = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/setA_less/99/'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pos_images = [os.path.join(pos_dir, img) for img in os.listdir(pos_dir)]
    neg_images = [os.path.join(neg_dir, img) for img in os.listdir(neg_dir)]

    pos_images.sort()
    neg_images.sort()

    anchore_paths = list()

    for i in range(0, len(pos_images) // 2):
        anchore_paths.append(pos_images.pop(i))
    # anchore_path_1 = pos_images.pop(0)
    # anchore_path_2 = pos_images.pop(-1)
    # pos_images.pop(-1)
    pos_images = pos_images[0: len(anchore_paths)]
    neg_images = neg_images[0: len(pos_images)]

    arcface = head_models.ArcFace(2048, 512, device_id=[torch.cuda._get_device_index(device)])
    # sphereface = head_models.SphereFace(2048, 512, device_id=[torch.cuda._get_device_index(device)])
    backbone = resnet.Resnet_152(512)
    head = siamasenet.SiamaseNet(device=device, head=arcface, head_name=arcface.name)

    backbone.to(device)
    head.to(device)
    backbone.eval()
    head.eval()

    transformations = transforms.Compose([
        transforms.CenterCrop(112),
        transforms.Resize(112),
        transforms.ToTensor()
    ])
    anchore_batch = torch.zeros((len(anchore_paths), 3, 112, 112))

    for i, img in enumerate(anchore_paths):
        image = Image.open(img)
        anchore_image = transformations(image)
        anchore_batch[i] = anchore_image

    # anchore_image_1 = Image.open(anchore_path_1)
    # anchore_image_1 = transformations(anchore_image_1)
    # # anchore_image_1 = anchore_image_1.to(device)
    #
    # anchore_image_2 = Image.open(anchore_path_2)
    # anchore_image_2 = transformations(anchore_image_2)
    # anchore_image_2 = anchore_image_2.to(device)

    # anchore_batch[0] = anchore_image_1
    # anchore_batch[1] = anchore_image_2

    anchore_batch = anchore_batch.to(device)

    anchore_features = backbone(anchore_batch)

    pos_batch = torch.zeros((len(pos_images), 3, 112, 112))
    neg_batch = torch.zeros((len(neg_images), 3, 112, 112))
    for i, img in enumerate(pos_images):
        image = Image.open(img)
        pos_img = transformations(image)
        pos_batch[i] = pos_img

    for i, img in enumerate(neg_images):
        image = Image.open(img)
        neg_img = transformations(image)
        neg_batch[i] = neg_img

    pos_batch = pos_batch.to(device)
    neg_batch = neg_batch.to(device)

    pos_features = backbone(pos_batch)
    neg_features = backbone(neg_batch)

    positive_distance, negative_distance = head(anchore_features, pos_features, neg_features)

    print(positive_distance, negative_distance)
