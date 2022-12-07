import resnet
from torchvision.models import ResNet152_Weights
from siamasenet import SiamaseNet
from head_models import ArcFace
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
input_features = 2048
output_features = 512
device_id = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

a = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)
p = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)
n = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)

p_label = torch.zeros(10)
n_label = torch.ones(10)
#
# pred = model(x)
#
# print(pred.shape)


model = resnet.Resnet_152(512)
model.to(device)
# print(model)

arcface = ArcFace(input_features, output_features)

# model.load_state_dict(ResNet152_Weights.get_state_dict(ResNet152_Weights.DEFAULT, True))
model.eval()

simesnet = SiamaseNet(device=device)
simesnet.to(device)
#
simesnet.eval()

# a.cuda(device)
# p.to(device)
# n.to(device)

print("is cuda: ", a.is_cuda)
print("is model cuda: ", torch.cuda.get_device_name(0))

a_features = model(a)
p_features = model(p)
n_features = model(n)

# print(a_features.size())
# print(a_features.data.shape)
# a_features.to(device)
# a_features = simesnet.flat(a_features)

# a_face = arcface(a_features, p_label)
# p_face = arcface(p_features, p_label)
# n_face = arcface(n_features, n_label)

# a_flat, p_flat, n_flat = simesnet(a_features, p_features, n_features)
#
# print(a_flat.shape)

ap_dis, an_dis = simesnet(a_features, p_features, n_features)
print(ap_dis, an_dis)

loss = torch.abs(ap_dis - an_dis)
print(f'sub: {loss}')
loss = torch.mean(torch.clamp(loss + 0.5, min=0.0))
print(loss)
#
# margin = 0.5
# loss = ap_dis - an_dis
# loss = torch.max(loss + margin)
# print(loss)
