import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import DataLoader

from model import resnet, siamasenet
from model.head_models import Softmax, SphereFace, CosFace, ArcFace, ShaoFace
from utils import get_latest_weights

from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="All Combination",
                                     description="Calculate and plot all combination for a given dataset.")
    # parser.add_argument('logFileName')
    parser.add_argument('backbonePath', nargs='?')
    parser.add_argument('headPath', nargs='?')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/frvt_detected_512/'
    weights_root = './weights'
    LOG_DIR = './logs'

    input_size = 112
    batch_size = 4
    input_feature_size = 2048
    embedding_size = 512
    num_workers = 4
    pin_memory = True

    input_transforms = Compose([
        CenterCrop(input_size),
        Resize(input_size),
        ToTensor()
    ])

    # data loader
    dataset = ImageFolder(data_root, transform=input_transforms)
    dataloader = DataLoader(dataset, batch_size, num_workers=num_workers, shuffle=False, pin_memory=pin_memory)

    # === Load Model === #
    backbone = resnet.Resnet_152(embedding_size)
    # softmax = Softmax(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # sphereface = SphereFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # sphereface = SphereFace(input_feature_size, embedding_size)
    cosface = CosFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # arcface = ArcFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # shaoface = ShaoFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # head = siamasenet.SiamaseNet(device=device, head_name='Linear')
    head = siamasenet.SiamaseNet(device=device, head=cosface, head_name=cosface.name)

    if args.backbonePath and args.headPath:
        latest_backbone_path, latest_head_path = os.path.join(weights_root, args.backbonePath), os.path.join(
            weights_root, args.headPath)
    else:
        latest_backbone_path, latest_head_path = get_latest_weights(weights_root)

    print(f'Loading Backbone from {latest_backbone_path}')
    backbone.load_state_dict(torch.load(latest_backbone_path))
    backbone.to(device)
    print(f'Loading Head from {latest_head_path}')
    head.load_state_dict(torch.load(latest_head_path))
    head.to(device)
    print("Model load complete.")

    for param in head.parameters():
        param.requires_grad = False

    backbone.eval()
    head.eval()

    extracted_features = list()

    print('Extracting Features...')

    for images, labels in tqdm(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        init_features = backbone(images)
        features, labels = head.extract_features(init_features, labels)
        for i, f in enumerate(features):
            extracted_features.append((f, labels[i]))

    positive_distances = list()
    negative_distances = list()

    print(f'{len(extracted_features)} features extracted.')
    print('Calculating distances...')

    for f1, l1 in tqdm(extracted_features):
        for f2, l2 in extracted_features:
            if torch.equal(f1, f2):
                continue
            else:
                distance = (torch.max(f1 - f2), l1)
                if l1 == l2:
                    positive_distances.append(distance)
                else:
                    negative_distances.append(distance)

    px, nx = list(), list()
    py, ny = list(), list()

    for pd, l in positive_distances:
        px.append(pd.detach().cpu().numpy())
        py.append(l.detach().cpu().numpy())

    for nd, l in negative_distances:
        nx.append(nd.detach().cpu().numpy())
        ny.append(l.detach().cpu().numpy())

    plt.scatter(px, py, alpha=0.5)
    plt.scatter(nx, ny, alpha=0.5)
    plt.show()
