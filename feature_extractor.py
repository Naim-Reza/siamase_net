import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor

from model import resnet

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def flatten_list(l):
    return [item for sublist in l for item in sublist]


root_dir = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/test_set_1/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = ImageFolder(root_dir, transform=Compose([
    CenterCrop(112),
    Resize(112),
    ToTensor()
]))

dataLoader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

backbone = resnet.Resnet_152(512)

flat = torch.nn.Flatten()
# img_values, label_values = list(), list()
# for images, labels in tqdm(iter(dataLoader)):
#     embeddings = backbone(images)
#     embeddings = flat(embeddings)
#     mean_embeddings = list()
#
#     for emb in embeddings:
#         mean = torch.mean(emb)
#         mean_embeddings.append(mean)
#     img_values.append(mean_embeddings)
#     label_values.append(labels)
#
# img_values = flatten_list(img_values)
# label_values = flatten_list(label_values)
#
# plt.scatter(img_values, label_values, c=label_values, alpha=0.5)
# plt.show()
embeddings = np.zeros((len(dataLoader.dataset), 2048))
labels = np.zeros(len(dataLoader.dataset))
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
          '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
          '#bcbd22', '#17becf']
with torch.no_grad():
    backbone.eval()
    k = 0
    for images, target in tqdm(dataLoader):
        # if cuda:
        #     images = images.cuda()
        emb = flat(backbone(images))
        embeddings[k:k + len(images)] = emb.data.cpu().numpy()
        labels[k:k + len(images)] = target.numpy()
        k += len(images)

plt.figure(figsize=(10, 10))
for i in range(10):
    inds = np.where(labels == i)[0]
    plt.scatter(embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i])
    # if xlim:
    #     plt.xlim(xlim[0], xlim[1])
    # if ylim:
    #     plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
plt.show()
