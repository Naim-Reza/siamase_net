import sys

sys.path.append('/media/naim/4A62E7E862E7D6AB/Users/chosun/siamase_net/')
import torch.utils.data
from model import resnet
from utils import get_latest_weights
from DataLoader import SiameseDataset
from sphere_model import SphereModel

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
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/test_set/'
    weights_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/siamase_net/weights'
    LOG_DIR = '/media/naim/4A62E7E862E7D6AB/Users/chosun/siamase_net/logs'

    input_size = 112
    batch_size = 4
    input_feature_size = 2048
    embedding_size = 2048
    num_workers = 4
    pin_memory = True

    # ==== DataLoader === #
    test_dataset = SiameseDataset(data_root, image_size=input_size, test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

    # === Load Model === #
    backbone = resnet.Resnet_152(embedding_size)
    head = SphereModel(input_feature_size, embedding_size)

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
    prev_class_name = 13340
    positive_dict = {}
    negative_dict = {}
    with torch.no_grad():
        for a, p, n, class_names, neg_labels in tqdm(test_dataloader, total=len(test_dataloader)):
            pos_labels, neg_labels = torch.tensor(class_names, dtype=torch.int64), torch.tensor(neg_labels,
                                                                                                dtype=torch.int64)
            anchore, positive, negative, pos_labels, neg_labels = a.to(device), p.to(device), n.to(
                device), pos_labels.to(device), neg_labels.to(device)
            anchore_features, positive_features, negative_features = backbone(anchore), backbone(positive), backbone(
                negative)

            positive_distance, negative_distance = head(anchore_features, positive_features, negative_features,
                                                        pos_labels, neg_labels)
            pds = positive_distance.detach().cpu().numpy()
            nds = negative_distance.detach().cpu().numpy()
            # print(class_names)

            for i, class_name in enumerate(class_names):
                # if int(class_name) == 98 or int(class_name) == 99:
                #     print(f'classname found: {class_name}')
                if not int(class_name) == int(prev_class_name):
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
