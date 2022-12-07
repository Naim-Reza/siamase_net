import torch.nn as nn
import torch


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SiamaseNet(nn.Module):
    def __init__(self):
        super(SiamaseNet, self).__init__()
        self.flat = Flatten()
        self.sc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=256)
        )
        self.distance = DistanceLayer()

    def forward(self, anchor, positive, negative):
        an_emb = self.sc(self.flat(anchor))
        pos_emb = self.sc(self.flat(positive))
        neg_emb = self.sc(self.flat(negative))
        return self.distance(an_emb, pos_emb, neg_emb)
        # return an_emb, pos_emb, neg_emb


class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # ap_distance = torch.sum(torch.square(anchor_embedding - positive_embedding))
        # an_distance = torch.sum(torch.square(anchor_embedding - negative_embedding))
        ap_distance = torch.max(torch.abs(anchor_embedding - positive_embedding), dim=1).values
        an_distance = torch.max(torch.abs(anchor_embedding - negative_embedding), dim=1).values
        return ap_distance, an_distance
