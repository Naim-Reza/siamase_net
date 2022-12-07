import resnet
from torchvision.models import ResNet152_Weights
from siamasenet import SiamaseNet
import torch

# input_size = [224, 224]
# mod = resnet.ResNet_152(input_size)
# mod.load_state_dict(ResNet152_Weights.get_state_dict(ResNet152_Weights.DEFAULT, True))
# print(mod)
#
# mod.eval()
#
# x = torch.randint_like(torch.zeros(10, 3, 112, 112), 0, 255)
#
# pred = mod(x)
#
# print(pred)

# model = resnet152(pretrained=True)
# print(model)
#
# model.eval()
#
a = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255)
p = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255)
n = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255)
#
# pred = model(x)
#
# print(pred.shape)


model = resnet.Resnet_152(512)
print(model)

# model.load_state_dict(ResNet152_Weights.get_state_dict(ResNet152_Weights.DEFAULT, True))
model.eval()

simesnet = SiamaseNet()

simesnet.eval()

a_features = model(a)
p_features = model(p)
n_features = model(n)

print(a_features.size())
print(a_features.data.shape)

ap_dis, an_dis = simesnet(a_features, p_features, n_features)
print(ap_dis, an_dis)

margin = 0.5
loss = ap_dis - an_dis
loss = torch.max(loss + margin)
print(loss)
