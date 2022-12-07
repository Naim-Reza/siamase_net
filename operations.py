from datetime import datetime
import os

import torch
from tqdm import tqdm
import numpy as np

LOG_DIR = './logs'
BACKBONE_NAME = 'Resnet152'
HEAD_NAME = 'ArcFace'


def train(train_dataloader, val_dataloader, backbone, head, num_epoch, device, criterion, optimizer, batch_size,
          weight_root):
    best_loss = 1000.00
    training_losses = list()
    validation_losses = list()

    writing_freq = 4

    for epoch in range(num_epoch):
        print(f'EPOCH: {epoch + 1}/{num_epoch}')
        print('=' * 20)

        backbone.to(device)
        head.to(device)

        backbone.eval()
        head.train()

        running_loss = 0.0

        for a, p, n in tqdm(iter(train_dataloader)):
            anchore, positive, negative = a.to(device), p.to(device), n.to(device)

            features_a, features_p, features_n = backbone(anchore), backbone(positive), backbone(negative)
            # print(features_a.shape, features_p.shape, features_n.shape)
            # a_emb, p_emb, n_emb = head(features_a, features_p, features_n)
            ap_dis, an_dis = head(features_a, features_p, features_n)
            # print(ap_dis, an_dis)

            loss = criterion.compute_loss(ap_dis, an_dis)
            criterion.update(loss)
            # print(torch.zeros_like(ap_dis))

            # loss = criterion(a_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # exp_lr_scheduler.step()

            running_loss += loss.item() * batch_size

            # print(f'Loss: {loss}, Running Loss: {running_loss}')

        epoch_loss = running_loss / len(train_dataloader)
        training_losses.append(epoch_loss)
        print(f'Epoch Loss: {epoch_loss}')

        print('Performing validation...')
        backbone.eval()
        head.eval()

        current_loss = 0.0
        for a, p, n in tqdm(iter(val_dataloader)):
            with torch.set_grad_enabled(False):
                anchore, positive, negative = a.to(device), p.to(device), n.to(device)
                features_a, features_p, features_n = backbone(anchore), backbone(positive), backbone(negative)
                # a_emb, p_emb, n_emb = head(features_a, features_p, features_n)
                ap_dis, an_dis = head(features_a, features_p, features_n)

                loss = criterion.compute_loss(ap_dis, an_dis)
                # loss = criterion(a_emb, p_emb, n_emb)
                current_loss += loss.item() * batch_size

        val_loss = current_loss / len(val_dataloader)
        validation_losses.append(val_loss)
        print(f'Validation Loss: {val_loss}')
        if best_loss > val_loss:
            print(f"Saving Weights for Epoch: {epoch + 1}")
            best_loss = val_loss
            torch.save(backbone.state_dict(),
                       os.path.join(weight_root, f"Backbone_{BACKBONE_NAME}_Epoch_{epoch + 1}_Time{get_time()}"))
            torch.save(head.state_dict(),
                       os.path.join(weight_root, f"Head_{HEAD_NAME}_Epoch_{epoch + 1}_Time{get_time()}"))

        if epoch % writing_freq == 0:
            writeable_array = np.array([training_losses, validation_losses])
            log_file = os.path.join(LOG_DIR, f'train_{HEAD_NAME}_log')
            np.save(log_file, writeable_array)

    return training_losses, validation_losses


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')
