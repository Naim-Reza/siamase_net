import torch.nn as nn
import torch

from model import sphere_face


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class SphereModel(nn.Module):
    def __init__(self, input_features=2048, output_features=512):
        super(SphereModel, self).__init__()
        self.head_name = 'SphereFace_V2'
        self.input_features = input_features
        self.output_features = output_features
        self.flat = Flatten()
        self.distance = DistanceLayer()
        self.head = sphere_face.AngleLinear(self.input_features, self.output_features, phiflag=True)
        self.head_out = sphere_face.AngleLoss()

    def forward(self, anchor, positive, negative, positive_labels, negative_labels):
        # positive_labels = torch.ones(anchor.shape[0], dtype=torch.int64).cuda()
        # negative_labels = torch.zeros(negative.shape[0], dtype=torch.int64).cuda()
        an_emb = self.head_out(self.head(self.flat(anchor)), positive_labels)
        pos_emb = self.head_out(self.head(self.flat(positive)), positive_labels)
        neg_emb = self.head_out(self.head(self.flat(negative)), negative_labels)

        return self.distance(an_emb, pos_emb, neg_emb)

    def extract_features(self, images, labels):
        return self.head(self.flat(images), labels), labels


class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # ap_distance = torch.sum(torch.square(anchor_embedding - positive_embedding))
        # an_distance = torch.sum(torch.square(anchor_embedding - negative_embedding))
        ap_distance = torch.max(torch.abs(anchor_embedding - positive_embedding), dim=1).values
        an_distance = torch.max(torch.abs(anchor_embedding - negative_embedding), dim=1).values
        return ap_distance, an_distance
