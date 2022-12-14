import resnet
# from torchvision.models import ResNet152_Weights
from siamasenet import SiamaseNet
from maam import MAAM
# from head_models import ArcFace, CosFace, SphereFace, Softmax, ShaoFace
import torch
import sphere_face
from PIL import Image


def printgradvals(module, grad_input, grad_output):
    # print(grad_input[0].ne(grad_input[0]).any())
    # print(grad_output[0].ne(grad_output[0]).any())
    # print(grad_input[0].abs().mean())
    # print(grad_output[0].abs().mean())
    # print(grad_input[0].abs().min())
    # print(grad_output[0].abs().min())
    print(grad_input)
    print(grad_output)


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    print(torch.__version__)
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
    # input_features = 2048
    # output_features = 512
    # device_id = [0]
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # a = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)
    # p = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)
    # n = torch.randint_like(torch.zeros(10, 3, 224, 224), 0, 255, device=device)
    #
    # p_label = torch.zeros(10)
    # n_label = torch.ones(10)
    # #
    # # pred = model(x)
    # #
    # # print(pred.shape)
    #
    #
    model = resnet.Resnet_152(512)
    maam_net = MAAM(2048, 512, m=5)
    maam_net.register_full_backward_hook(printgradvals)
    # sp_face = sphere_face.AngleLinear(2048, 512, phiflag=False)
    # sp_face_loss = sphere_face.AngleLoss()
    # model.to(device)
    # # print(model)
    #
    # softmax = Softmax(input_features, output_features, device_id=[torch.cuda._get_device_index(device)])
    # sphereface = SphereFace(input_features, output_features, device_id=[torch.cuda._get_device_index(device)])
    # cosface = CosFace(input_features, output_features, device_id=[torch.cuda._get_device_index(device)])
    # arcface = ArcFace(input_features, output_features, device_id=[torch.cuda._get_device_index(device)])
    # shaoface = ShaoFace(input_features, output_features, device_id=[torch.cuda._get_device_index(device)])
    #
    # # model.load_state_dict(ResNet152_Weights.get_state_dict(ResNet152_Weights.DEFAULT, True))
    # model.eval()
    #
    simesnet = SiamaseNet(head_name='Linear')
    # # simesnet = SiamaseNet(device=device, head=softmax, head_name=softmax.name)
    # # simesnet = SiamaseNet(device=device, head=sphereface, head_name=sphereface.name)
    # # simesnet = SiamaseNet(device=device, head=cosface, head_name=cosface.name)
    # simesnet = SiamaseNet(device=device, head=shaoface, head_name=shaoface.name)
    # simesnet.to(device)
    # #
    # simesnet.eval()
    #

    a_imgs = torch.randint_like(torch.zeros(4, 3, 224, 224), 0, 255)
    imgs = torch.randint_like(torch.zeros(4, 3, 224, 224), 0, 255)
    targets = torch.randint_like(torch.zeros(4, 1), 0, 4, dtype=torch.int64)
    model.eval()
    # sp_face.eval()
    maam_net.eval()
    optimizer = torch.optim.Adam(maam_net.parameters(), lr=1e-3)

    # with torch.no_grad():
    #     a_features = model(a_imgs)
    #     features = model(imgs)
    #     a_features = simesnet.flat(a_features)
    #     features = simesnet.flat(features)
    #     a_sp_features = sp_face(features)
    #     sp_features = sp_face(features)
    #     a_head_features = sp_face_loss(a_sp_features, targets)
    #     head_features = sp_face_loss(sp_features, targets)
    #
    #     distance = torch.sqrt(a_head_features ** 2 + head_features ** 2)
    #     print(torch.max(torch.abs(a_head_features - head_features), dim=1).values)

    # with torch.no_grad():
    a_features = model(a_imgs)
    features = model(imgs)
    a_features = simesnet.flat(a_features)
    features = simesnet.flat(features)

    a_head_features = maam_net(a_features, targets)
    img_head_features = maam_net(features, targets)

    # distance = torch.sqrt(a_head_features ** 2 + head_features ** 2)
    print(torch.max(a_head_features - img_head_features, dim=1).values)
    loss = torch.max(a_head_features - img_head_features, dim=1).values
    loss = torch.max(torch.clamp(loss + 0.5, min=0.0))
    print(loss)
    optimizer.zero_grad()
    loss.backward()
