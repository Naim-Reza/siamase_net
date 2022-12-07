import torch.nn as nn
from torchvision.models import resnet152


# class Resnet(nn.Module):
#     def __int__(self):
#         super(Resnet, self).__int__()
#         self.resnet_152 = resnet152()
#
#     def forward(self, x):
#         x = self.resnet_152(x)
#         return x


def Resnet_152(out_features):
    m = resnet152(pretrained=True)
    modules = list(m.children())[:-1]
    m = nn.Sequential(*modules)
    for param in m.parameters():
        param.requires_grad = False

    # n_features = m.fc.in_features
    # m.fc = nn.Linear(n_features, out_features)
    return m
