import torch.nn as nn
import torch
from head_models import ArcFace


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SiamaseNet(nn.Module):
    def __init__(self, head=None, head_name='Linear', input_features=2048, output_features=512, device=None):
        super(SiamaseNet, self).__init__()
        assert head_name in ['ShaoFace', 'ArcFace', 'CosFace', 'SphereFace', 'SoftMax', 'Linear']
        self.head_name = head_name
        self.input_features = input_features
        self.output_features = output_features
        self.device = device
        self.arcface = ArcFace(self.input_features, self.output_features, device_id=[0])
        self.flat = Flatten()
        self.lin = nn.Linear(in_features=self.input_features, out_features=self.output_features)
        self.sc = nn.Sequential(
            nn.Linear(in_features=self.input_features, out_features=self.output_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=512),
            nn.Linear(in_features=512, out_features=512)
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256),
            # nn.Linear(in_features=256, out_features=256)
        )
        self.distance = DistanceLayer()
        if self.head_name == 'Linear':
            self.head = self.sc
        else:
            self.head = head

        self.requires_label = ['ShaoFace', 'ArcFace', 'CosFace', 'SphereFace']

    def forward(self, anchor, positive, negative):
        if self.head_name not in self.requires_label:
            an_emb = self.sc(self.flat(anchor))
            pos_emb = self.sc(self.flat(positive))
            neg_emb = self.sc(self.flat(negative))
            return self.distance(an_emb, pos_emb, neg_emb)

        positive_labels = torch.ones(anchor.shape[0]).to(self.device)
        negative_labels = torch.zeros(negative.shape[0]).to(self.device)
        an_emb = self.head(self.flat(anchor), positive_labels)
        pos_emb = self.head(self.flat(positive), positive_labels)
        neg_emb = self.head(self.flat(negative), negative_labels)

        # an_emb = self.lin(self.flat(anchor))
        # pos_emb = self.lin(self.flat(positive))
        # neg_emb = self.lin(self.flat(negative))

        # an_emb = self.sc(self.flat(anchor))
        # pos_emb = self.sc(self.flat(positive))
        # neg_emb = self.sc(self.flat(negative))
        return self.distance(an_emb, pos_emb, neg_emb)
        # return an_emb, pos_emb, neg_emb
        # return self.flat(anchor), self.flat(positive), self.flat(negative)


class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # ap_distance = torch.sum(torch.square(anchor_embedding - positive_embedding))
        # an_distance = torch.sum(torch.square(anchor_embedding - negative_embedding))
        ap_distance = torch.max(torch.abs(anchor_embedding - positive_embedding), dim=1).values
        an_distance = torch.max(torch.abs(anchor_embedding - negative_embedding), dim=1).values
        return ap_distance, an_distance
