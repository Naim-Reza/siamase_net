import torch.utils.data
from model import resnet, siamasenet
from model.head_models import Softmax, SphereFace, CosFace, ArcFace, ShaoFace
from utils import get_latest_weights
from DataLoader import SiameseDataset

import os
import numpy as np
from tqdm import tqdm
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Test", description="Generate test result for trained model")
    parser.add_argument('logFileName')
    parser.add_argument('backbonePath', nargs='?')
    parser.add_argument('headPath', nargs='?')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/frvt_detected_faces/'
    weights_root = './weights'
    LOG_DIR = './logs'

    input_size = 112
    batch_size = 4
    input_feature_size = 2048
    embedding_size = 512
    num_workers = 4
    pin_memory = True

    # ==== DataLoader === #
    test_dataset = SiameseDataset(data_root, image_size=input_size, test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

    # === Load Model === #
    backbone = resnet.Resnet_152(embedding_size)
    # softmax = Softmax(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    sphereface = SphereFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # cosface = CosFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # arcface = ArcFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # shaoface = ShaoFace(input_feature_size, embedding_size, device_id=[torch.cuda._get_device_index(device)])
    # head = siamasenet.SiamaseNet(device=device, head_name='Linear')
    head = siamasenet.SiamaseNet(device=device, head=sphereface, head_name=sphereface.name)

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

    # === Perform Testing === #
    positive_distances = list()
    negative_distances = list()
    prev_class_name = 'S518'
    positive_dict = {}
    negative_dict = {}
    for a, p, n, class_names in tqdm(test_dataloader, total=len(test_dataloader)):
        anchore, positive, negative = a.to(device), p.to(device), n.to(device)
        with torch.no_grad():
            anchore_features, positive_features, negative_features = backbone(anchore), backbone(positive), backbone(
                negative)

            positive_distance, negative_distance = head(anchore_features, positive_features, negative_features)
            pds = positive_distance.detach().cpu().numpy()
            nds = negative_distance.detach().cpu().numpy()
        # print(class_names)

        for i, class_name in enumerate(class_names):
            # if int(class_name) == 98 or int(class_name) == 99:
            #     print(f'classname found: {class_name}')
            if not class_name == prev_class_name:
                # print(f'Previous class: {prev_class_name}, Current class: {class_name}')
                positive_dict[prev_class_name] = positive_distances
                negative_dict[prev_class_name] = negative_distances
                prev_class_name = class_name
                positive_distances = []
                negative_distances = []

            positive_distances.append(pds[i])
            negative_distances.append(nds[i])
            positive_dict[class_name] = positive_distances
            negative_dict[class_name] = negative_distances

        # print(positive_distances)

    log_file = os.path.join(LOG_DIR, f'test_{args.logFileName}_logs.npy')
    print(f'saving test result logs in {log_file}')

    np.save(log_file, np.array([positive_dict, negative_dict]))
    print(f'Test log saved in {log_file}')

    # print(positive_dict)
    # print(negative_dict)
