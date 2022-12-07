from itertools import chain

import torch
import torch.optim as optim
import torch.utils.data

from DataLoader import SiameseDataset
from loss import TripletLoss
from model import siamasenet, resnet
from operations import train
from utils import show_training_results

if __name__ == '__main__':
    # ===== Training Configuration ===  #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # train_data_root = ''
    # anchor_data_root = ''
    # positive_data_root = ''
    data_root = '/media/naim/4A62E7E862E7D6AB/Users/chosun/Datasets/setA_less/'

    input_size = 112
    batch_size = 4
    embedding_size = 512
    num_workers = 4
    pin_memory = True

    lr = 1e-4
    momentum = 0.3
    weight_decay = 2e-4
    decay_step = 4
    gamma = 0.1
    epochs = 100

    # ====== Data Loader ===== #
    train_dataset = SiameseDataset(data_root, input_size)
    val_dataset = SiameseDataset(data_root, input_size, train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=num_workers,
                                                   pin_memory=pin_memory)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size, num_workers=num_workers,
                                                 pin_memory=pin_memory)

    # ===== Embedding generator ==== #
    backbone = resnet.Resnet_152(embedding_size)
    head = siamasenet.SiamaseNet(device=device)

    # def distance_function(x1, x2):
    #     return torch.max(torch.abs(x1 - x2), dim=1).values

    # ==== Training ==== #
    loss_margin = 0.5
    model_parameters = chain(backbone.parameters(), head.parameters())
    criterion = TripletLoss(loss_margin)
    # criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=loss_margin)
    # optimizer = optim.SGD(model_parameters, lr=lr, momentum=momentum)
    optimizer = optim.Adam(model_parameters, lr=lr)
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=gamma)
    training_losses, validation_losses = train(train_dataloader, val_dataloader, backbone, head, epochs, device,
                                               criterion, optimizer, batch_size, weight_root='./weights')

    print("=" * 30)
    print("Training Completed..")
    show_training_results(training_losses, validation_losses)
