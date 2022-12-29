import math

import torch
import torch.nn as nn


class MAAM(nn.Module):
    def __init__(self, input_features, out_features, m=4):
        super(MAAM, self).__init__()
        self.name = 'MAAM'
        self.input_features = input_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, input_features))
        self.m = m
        self.gama = 0.1
        self.lamda_max = 1000.0
        self.lamda_min = 5.0
        self.lamda = 1000.0
        self.iteration = 0
        self.threshold = math.cos((math.pi - m) / m)
        self.mm = math.sin((math.pi - m) / m) * m

        # self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        nn.init.xavier_uniform_(self.weight)

        self.fn_m = [
            lambda x, y: x ** 0,
            lambda x, y: x * math.cos(self.m) - y * math.sin(self.m),
            lambda x, y: (2 * x ** 2 - 1) * math.cos(self.m) - (2 * y * x) * math.sin(self.m),
            lambda x, y: (4 * x ** 3 - 3 * x) * math.cos(self.m) - (3 * y - 4 * y ** 3) * math.sin(self.m),
            lambda x, y: (8 * x ** 4 - 4 * x ** 2 + 1) * math.cos(self.m) - (
                    2 * (2 * y * x) * (2 * x ** 2 - 1)) * math.sin(self.m),
            lambda x, y: ((4 * x ** 3 - 3 * x) * (2 * x ** 2 - 1) - (3 * y - 4 * y ** 3) * (2 * y * x)) * math.cos(
                self.m) - ((3 * y - 4 * y ** 3) * (2 * x ** 2 - 1) + (4 * x ** 3 - 3 * x) * (2 * y * x)) * math.sin(
                self.m),
            lambda x, y: ((4 * x ** 3 - 3 * x) ** 2 - (3 * y - 4 * y ** 3) ** 2) * math.cos(self.m) - 2 * (
                    (3 * y - 4 * y ** 3) * x) * math.sin(self.m)
        ]

    def forward(self, input, target):
        # self.iteration += 1
        x = input
        # w = self.weight
        #
        x_len = x.pow(2).sum(1).pow(0.5)
        #
        # ww = w.renorm(2, 1, 1e-5).mul(1e5)

        cos_theta = nn.functional.linear(nn.functional.normalize(x), nn.functional.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2))
        theta = cos_theta.data.acos()
        k = ((self.m * theta) / torch.pi).floor()

        cos_m_theta_plus_m = self.fn_m[self.m](cos_theta, sin_theta)
        cos_m_theta_plus_m = torch.where(cos_theta > self.threshold, cos_m_theta_plus_m, cos_theta - self.mm)

        n_one = k * 0.0 - 1
        phi_theta = (n_one ** k) * cos_m_theta_plus_m - 2 * k

        cos_m_theta_plus_m = cos_m_theta_plus_m * x_len.view(-1, 1)
        phi_theta = phi_theta * x_len.view(-1, 1)
        # print(f'Phi_theta: {phi_theta}')
        # print(f'CosMThetaPlusM: {cos_m_theta_plus_m}')

        target = target.view(-1, 1)
        index = torch.zeros_like(cos_m_theta_plus_m)
        index.scatter_(1, target.data.view(-1, 1), 1)
        index = index.byte()
        # index = torch.autograd.Variable(index)

        # self.lamda = max(self.lamda_min, self.lamda_max / (1 + self.gama * self.iteration))

        output = torch.ones_like(cos_m_theta_plus_m)
        output[index] = phi_theta[index] - cos_m_theta_plus_m[index]

        return output
