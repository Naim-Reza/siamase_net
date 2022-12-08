import torch


class TripletLoss:
    def __init__(self, margin):
        self.margin = margin
        self.loss_val = 0

    def compute_loss(self, ap_dis, an_dis):
        # loss = torch.abs(ap_dis - an_dis)
        loss = ap_dis - an_dis
        return torch.max(torch.clamp(loss + self.margin, min=0.0))

    def update(self, new_val):
        self.loss_val = new_val
