import torch.utils.data
from model import resnet, siamasenet
from utils import get_latest_weights
from DataLoader import SiameseDataset

import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/test_set/'
    weights_root = './weights'
    LOG_DIR = './logs'

    input_size = 112
    batch_size = 4
    embedding_size = 512
    num_workers = 4
    pin_memory = True

    # ==== DataLoader === #
    test_dataset = SiameseDataset(data_root, image_size=input_size, test=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size, num_workers=num_workers,
                                                  pin_memory=pin_memory)

    # === Load Model === #
    backbone = resnet.Resnet_152(embedding_size)
    head = siamasenet.SiamaseNet()

    latest_backbone_path, latest_head_path = get_latest_weights(weights_root)

    print(f'Loading Backbone from {latest_backbone_path}')
    backbone.load_state_dict(torch.load(latest_backbone_path))
    backbone.to(device)
    print(f'Loading Head from {latest_head_path}')
    head.load_state_dict(torch.load(latest_head_path))
    head.to(device)
    print("Model load complete.")

    backbone.eval()
    head.eval()

    # === Perform Testing === #
    positive_distances = list()
    negative_distances = list()
    prev_class_name = 0
    positive_dict = {}
    negative_dict = {}
    for a, p, n, class_names in tqdm(iter(test_dataloader), total=len(test_dataloader)):
        anchore, positive, negative = a.to(device), p.to(device), n.to(device)
        anchore_features, positive_features, negative_features = backbone(anchore), backbone(positive), backbone(
            negative)

        positive_distance, negative_distance = head(anchore_features, positive_features, negative_features)
        pds = positive_distance.detach().cpu().numpy()
        nds = negative_distance.detach().cpu().numpy()

        for i, class_name in enumerate(class_names):
            if not int(class_name) == prev_class_name:
                # print(f'Previous class: {prev_class_name}, Current class: {class_name}')
                positive_dict[prev_class_name] = positive_distances
                negative_dict[prev_class_name] = negative_distances
                prev_class_name = int(class_name)
                positive_distances = []
                negative_distances = []

            positive_distances.append(pds[i])
            negative_distances.append(nds[i])

        # print(positive_distances)

    log_file = os.path.join(LOG_DIR, 'test_200_logs.npy')
    print(f'saving test result logs in {log_file}')

    np.save(log_file, np.array([positive_dict, negative_dict]))
    print(f'Test log saved in {log_file}')

    # print(positive_dict)
    # print(negative_dict)
