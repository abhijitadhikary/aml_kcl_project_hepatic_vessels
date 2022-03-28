import os
import time
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
import nibabel as nib
from tqdm import tqdm
import json
import cv2
import copy
import torch.nn.functional as F

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# from dataloader_top_left import DatasetHepatic
from dataloader_center import DatasetHepatic

# from network_resnet_deepmedic import DeepMedic
from network_deepmedic import DeepMedic

from losses import GeneralizedDiceLoss

from stride_checker import stride_depth_and_inference

patch_size_normal = 25
patch_size_low = 19
patch_size_out = 9
patch_low_factor = 3

# either batch_size or batch_size_inner MUST be set to 1
batch_size = 1
batch_size_inner = 16
run_mode = 'train'

# run inference with batch size = 1
if run_mode == 'inference':
    batch_size = 1

dataset = DatasetHepatic(
    run_mode=run_mode,
    label_percentage=0.0001,
    use_probabilistic=True,
    patch_size_normal=patch_size_normal,
    patch_size_low=patch_size_low,
    patch_size_out=patch_size_out,
    patch_low_factor=patch_low_factor,
    create_numpy_dataset=False,
    dataset_variant='npy',
    batch_size_inner=batch_size_inner
)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False
)

# for i in range(20):
#     batch = next(iter(dataloader))
#     print(f'{i}\t{batch[0].shape}\t{batch[1].shape}\t{batch[2].shape}')


learning_rate = 0.0001
momentum = 0.9
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = DeepMedic().to(device)

# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    amsgrad=True
)

criterion_mse = nn.MSELoss()
# criterion_mse = get_mse_loss
criterion_dice = GeneralizedDiceLoss().dice
criterion_ce = nn.CrossEntropyLoss()
num_epochs = 10

for index_epoch in tqdm(range(num_epochs), leave=False):
    time_start = time.time()
    loss_list_mse = []
    for index_batch, batch in enumerate(dataloader):
        if run_mode == 'train':
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            model.train()
            optimizer.zero_grad()
            image_patch_normal, image_patch_low, label_patch_out_real = batch
            image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.to(
                device), image_patch_low.to(device), label_patch_out_real.to(device)
            if batch_size_inner > 1:
                image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.squeeze(
                    0), image_patch_low.squeeze(0), label_patch_out_real.squeeze(0)
            # forward pass
            label_patch_out_pred = model.forward((image_patch_normal, image_patch_low))

            # cross-entropy loss
            # loss = criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1).long())

            # convert label_patch_out_real to one hot
            label_patch_out_real_one_hot = torch.zeros_like(label_patch_out_pred).to(device)
            # label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 0, 1)
            # label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 1)
            # label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 2, 1)

            label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 1, 0).squeeze(1)
            label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 0).squeeze(1)
            label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 1, 0).squeeze(1)

            # generalized dice loss
            loss = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)

            # calculate gradients and update weights
            loss.backward()
            optimizer.step()

            loss_list_mse.append(loss.item())
            print(loss.item())
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

        elif run_mode == 'inference':
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            labels_pred, model, optimizer, criterion_dice, loss = stride_depth_and_inference(model=model,
                                                                                             optimizer=optimizer,
                                                                                             criterion_dice=criterion_dice,
                                                                                             images_real=images,
                                                                                             labels_real=labels,
                                                                                             patch_size_normal=25,
                                                                                             patch_size_low=19,
                                                                                             patch_size_out=9,
                                                                                             patch_low_factor=3)

            # calculate loss

    loss_mse_epoch = sum(loss_list_mse) / len(loss_list_mse)

    duration = time.time() - time_start
    print(f'Epoch:\t[{index_epoch + 1} / {num_epochs}]\t\tTime:\t{duration} s\t\tMSE:\t{loss_mse_epoch:.5f}')
