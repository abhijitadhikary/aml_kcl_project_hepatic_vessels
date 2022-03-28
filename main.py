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




# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)




num_epochs = 10

def run_epoch(dataloader, model, optimizer, criterion_dice, criterion_mse, criterion_ce, device, run_mode):
    loss_list_dice = []
    loss_list_ce = []
    loss_list_mse = []

    for index_batch, batch in enumerate(dataloader):
        if run_mode == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            run_mode.eval()

        image_patch_normal, image_patch_low, label_patch_out_real = batch
        image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.to(device), image_patch_low.to(device), label_patch_out_real.to(device)
        if batch_size_inner > 1:
            image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.squeeze(0), image_patch_low.squeeze(0), label_patch_out_real.squeeze(0)

        # forward pass
        label_patch_out_pred = model.forward((image_patch_normal, image_patch_low))

        # cross-entropy loss
        # loss = criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1).long())

        # convert label_patch_out_real to one hot
        label_patch_out_real_one_hot = torch.zeros_like(label_patch_out_pred).to(device)

        label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 1, 0).squeeze(1)
        label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 0).squeeze(1)
        label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 1, 0).squeeze(1)

        # generalized dice loss
        loss_dice = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)
        loss_mse = criterion_mse(label_patch_out_pred.float(), label_patch_out_real_one_hot.float())

        if run_mode == 'train':
            # calculate gradients and update weights
            loss_dice.backward()
            optimizer.step()

        loss_list_dice.append(loss_dice.item())
        loss_list_mse.append(loss_mse)
        # print(loss_dice.item())
        # print(loss_mse.item())

    loss_dice_epoch = sum(loss_list_dice) / len(loss_list_dice)
    loss_mse_epoch = 0
    loss_ce_epoch = 0

    return model, optimizer, loss_dice_epoch, loss_mse_epoch, loss_ce_epoch

def fit_model(model_name, optimizer_name, dataloader_train, dataloader_val, learning_rate, num_epochs):

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    if model_name == 'deep_medic':
        model = DeepMedic().to(device)

    criterion_mse = nn.MSELoss()
    criterion_dice = GeneralizedDiceLoss().dice
    criterion_ce = nn.CrossEntropyLoss()


    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            amsgrad=True
        )
    elif optimizer_name == 'sgd_w_momentum':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    for index_epoch in tqdm(range(num_epochs), leave=False):
        time_start = time.time()

        loss_list_dice_train = []
        loss_list_dice_val = []
        loss_list_mse_train = []
        loss_list_mse_val = []

        # train
        model, optimizer, loss_dice_epoch, loss_mse_epoch, loss_ce_epoch = run_epoch(
            dataloader=dataloader_train,
            model=model,
            optimizer=optimizer,
            criterion_dice=criterion_dice,
            criterion_mse=criterion_mse,
            criterion_ce=criterion_ce,
            device=device,
            run_mode='train'
        )

        loss_list_dice_train.append(loss_dice_epoch)
        loss_list_mse_train.append(loss_mse_epoch)

        # validation
        with torch.no_grad():
            _, _, loss_dice_epoch, loss_mse_epoch, loss_ce_epoch = run_epoch(
                dataloader=dataloader_val,
                model=model,
                optimizer=optimizer,
                criterion_dice=criterion_dice,
                criterion_mse=criterion_mse,
                criterion_ce=criterion_ce,
                device=device,
                run_mode='val'
            )

        loss_list_dice_val.append(loss_dice_epoch)
        loss_list_mse_val.append(loss_mse_epoch)

        duration = time.time() - time_start
        print(f'Epoch:\t[{index_epoch + 1} / {num_epochs}]\t\tTime:\t{duration} s\t\tMSE:\t{loss_mse_epoch:.5f}')

fit_model(
    model_name='deep_medic',
    optimizer_name='adam',
    dataloader_train=dataloader,
    dataloader_val=dataloader,
    learning_rate=0.001,
    num_epochs=10
)


if run_mode == 'inference':
    for index_batch, batch in enumerate(dataloader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        labels_pred, model, optimizer, criterion_dice, loss = stride_depth_and_inference(
            model=model,
            optimizer=optimizer,
            criterion_dice=criterion_dice,
            images_real=images,
            labels_real=labels,
            patch_size_normal=patch_size_normal,
            patch_size_low=patch_size_low,
            patch_size_out=patch_size_out,
            patch_low_factor=patch_low_factor
        )



