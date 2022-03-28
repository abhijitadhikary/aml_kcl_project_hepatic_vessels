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
train_percentage = 0.8

def get_dataloaders(
        patch_size_normal,
        patch_size_low,
        patch_size_out,
        patch_low_factor,
        batch_size,
        batch_size_inner,
        train_percentage,

):
    dataset_train = DatasetHepatic(
        run_mode='train',
        label_percentage=0.0001,
        use_probabilistic=True,
        patch_size_normal=patch_size_normal,
        patch_size_low=patch_size_low,
        patch_size_out=patch_size_out,
        patch_low_factor=patch_low_factor,
        create_numpy_dataset=False,
        dataset_variant='npy',
        batch_size_inner=batch_size_inner,
        train_percentage=train_percentage
    )

    dataset_val = DatasetHepatic(
        run_mode='val',
        label_percentage=0.0001,
        use_probabilistic=True,
        patch_size_normal=patch_size_normal,
        patch_size_low=patch_size_low,
        patch_size_out=patch_size_out,
        patch_low_factor=patch_low_factor,
        create_numpy_dataset=False,
        dataset_variant='npy',
        batch_size_inner=batch_size_inner,
        train_percentage=train_percentage
    )

    dataset_inference = DatasetHepatic(
        run_mode='inference',
        label_percentage=0.0001,
        use_probabilistic=True,
        patch_size_normal=patch_size_normal,
        patch_size_low=patch_size_low,
        patch_size_out=patch_size_out,
        patch_low_factor=patch_low_factor,
        create_numpy_dataset=False,
        dataset_variant='npy',
        batch_size_inner=batch_size_inner,
        train_percentage=train_percentage
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True
    )

    dataloader_inference = DataLoader(
        dataset_inference,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader_train, dataloader_val, dataloader_inference


def early_stopper(loss_list_val, loss_least_val, early_stop_patience_counter, min_early_stop):
    '''
        Adapter from one of my previous implementations: https://github.com/abhijitadhikary/Twitter-Sentiment-Analysis-using-LSTM/blob/master/train.py
    '''
    valid_loss_list_cp = loss_list_val.copy()
    valid_loss_list_cp = valid_loss_list_cp[::-1]

    current_valid_loss = valid_loss_list_cp[0]
    if current_valid_loss < loss_least_val:
        loss_least_val = current_valid_loss
        early_stop_patience_counter = 0

    else:
        early_stop_condition = True
        for index in range(min_early_stop):
            if current_valid_loss < valid_loss_list_cp[index]:
                early_stop_condition = False
        if early_stop_condition is True:
            early_stop_patience_counter += 1
        else:
            early_stop_patience_counter = 0

    return loss_least_val, early_stop_patience_counter


def run_epoch(dataloader, model, optimizer, criterion_dice, criterion_mse, criterion_ce, device, run_mode, loss_mode):
    '''
        Runs one epoch of training and validation loops
    '''
    loss_list_dice = []
    loss_list_ce = []
    loss_list_mse = []
    loss_list_dice_n_mse = []

    for index_batch, batch in enumerate(dataloader):
        if run_mode == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            run_mode.eval()

        image_patch_normal, image_patch_low, label_patch_out_real = batch
        image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.to(device), image_patch_low.to(
            device), label_patch_out_real.to(device)
        if batch_size_inner > 1:
            image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.squeeze(
                0), image_patch_low.squeeze(0), label_patch_out_real.squeeze(0)

        # forward pass
        label_patch_out_pred = model.forward((image_patch_normal, image_patch_low))

        # cross-entropy loss
        # loss = criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1).long())

        # convert label_patch_out_real to one hot
        label_patch_out_real_one_hot = torch.zeros_like(label_patch_out_pred).to(device)

        label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 1, 0).squeeze(1)
        label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 0).squeeze(1)
        label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 1, 0).squeeze(1)

        # generalized dice loss # dice_only, mse_only, dice_n_mse
        if loss_mode == 'dice_only':
            loss_dice = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)
            loss_list_dice.append(loss_dice.item())
            loss = loss_dice
        if loss_mode == 'mse_only':
            loss_mse = criterion_mse(label_patch_out_pred.float(), label_patch_out_real_one_hot.float())
            loss_list_mse.append(loss_mse.item())
            loss = loss_mse
        if loss_mode == 'dice_n_mse':
            loss_dice = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)
            loss_mse = criterion_mse(label_patch_out_pred.float(), label_patch_out_real_one_hot.float())
            loss_dice_n_mse = loss_dice + loss_mse
            loss = loss_dice_n_mse
            loss_list_dice.append(loss_dice.item())
            loss_list_mse.append(loss_mse.item())
            loss_list_dice_n_mse.append(loss_dice_n_mse.item())

        if run_mode == 'train':
            # calculate gradients and update weights
            loss.backward()
            optimizer.step()

        print(loss_dice.item())
        # print(loss_mse.item())
        # print(loss_list_dice_n_mse.item())

    loss_dice = sum(loss_list_dice) / len(loss_list_dice)
    loss_mse = 0
    loss_dice_n_mse = 0

    return model, optimizer, loss_dice, loss_mse, loss_dice_n_mse

def fit_model(
        model_name,
        optimizer_name,
        learning_rate,
        num_epochs,
        patience_lr_scheduler,
        factor_lr_scheduler,
        patience_early_stop,
        min_early_stop,
        loss_mode,
        save_condition,
        resume_condition,
        resume_epoch,
        patch_size_normal=25,
        patch_size_low=19,
        patch_size_out=9,
        patch_low_factor=3,
        batch_size=1,
        batch_size_inner=16,
        train_percentage=0.8
    ):
    '''
        Trains and validates a model given hyperparameters, model and optimizer name, dataloaders, num_epochs
    '''

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')



    # get the dataloaders
    dataloader_train, dataloader_val, _ = get_dataloaders(
        patch_size_normal,
        patch_size_low,
        patch_size_out,
        patch_low_factor,
        batch_size,
        batch_size_inner,
        train_percentage
    )

    if model_name == 'deep_medic':
        model = DeepMedic().to(device)

    # loss funcitons
    criterion_mse = nn.MSELoss()
    criterion_dice = GeneralizedDiceLoss().dice
    criterion_ce = nn.CrossEntropyLoss()

    # choice of optimizer
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

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience_lr_scheduler,
                                                           factor=factor_lr_scheduler, verbose=True)

    early_stop_patience_counter = 0
    loss_best_dice = 1e12
    loss_best_mse = 1e12
    loss_best_dice_n_mse = 1e12

    if resume_condition:
        args, model, optimizer = load_model(resume_epoch, model, optimizer)

        save_condition, index_epoch, learning_rate, model, optimizer, early_stop_patience_counter, \
        loss_best_dice, loss_best_mse, loss_best_dice_n_mse, loss_list_dice_val, loss_list_mse_val, \
        loss_list_dice_n_mse_val, loss_list_dice_train, loss_list_mse_train, loss_list_dice_n_mse_train, \
        loss_mode, num_epochs, model_name, optimizer_name = args

    for index_epoch in tqdm(range(num_epochs), leave=False):
        time_start = time.time()

        loss_list_dice_train = []
        loss_list_dice_val = []
        loss_list_mse_train = []
        loss_list_mse_val = []
        loss_list_dice_n_mse_train = []
        loss_list_dice_n_mse_val = []

        # train
        model, optimizer, loss_dice, loss_mse, loss_dice_n_mse = run_epoch(
            dataloader=dataloader_train,
            model=model,
            optimizer=optimizer,
            criterion_dice=criterion_dice,
            criterion_mse=criterion_mse,
            criterion_ce=criterion_ce,
            device=device,
            run_mode='train',
            loss_mode=loss_mode
        )

        loss_list_dice_train.append(loss_dice)
        loss_list_mse_train.append(loss_mse)
        loss_list_dice_n_mse_train.append(loss_mse)

        # validation
        with torch.no_grad():
            _, _, loss_dice, loss_mse, loss_dice_n_mse = run_epoch(
                dataloader=dataloader_val,
                model=model,
                optimizer=optimizer,
                criterion_dice=criterion_dice,
                criterion_mse=criterion_mse,
                criterion_ce=criterion_ce,
                device=device,
                run_mode='val',
                loss_mode=loss_mode
            )

        loss_list_dice_val.append(loss_dice)
        loss_list_mse_val.append(loss_mse)
        loss_list_dice_n_mse_val.append(loss_dice_n_mse)

        # choose which loss to put into the scheduler
        scheduler.step(loss_dice)

        duration = time.time() - time_start
        print(f'Epoch:\t[{index_epoch + 1} / {num_epochs}]\t\tTime:\t{duration} s\t\tMSE:\t{loss_mse:.5f}')

        # loss list for early stopping
        args = save_condition, index_epoch, learning_rate, model, optimizer, early_stop_patience_counter, \
               loss_best_dice, loss_best_mse, loss_best_dice_n_mse, loss_list_dice_val, loss_list_mse_val, \
               loss_list_dice_n_mse_val, loss_list_dice_train, loss_list_mse_train, loss_list_dice_n_mse_train, \
               loss_mode, num_epochs, model_name, optimizer_name
        if loss_mode == 'dice_only':
            loss_best_dice = save_model(args, save_condition, index_epoch, loss=loss_dice, loss_best=loss_best_dice)
            loss_least_val = loss_best_dice
            loss_list_val = loss_list_dice_val
        elif loss_mode == 'mse_only':
            loss_best_mse = save_model(args, save_condition, index_epoch, loss=loss_mse, loss_best=loss_best_mse)
            loss_least_val = loss_best_mse
            loss_list_val = loss_list_mse_val
        elif loss_mode == 'dice_n_mse':
            loss_best_dice_n_mse = save_model(args, save_condition, index_epoch, loss=loss_dice_n_mse,
                                              loss_best=loss_best_dice_n_mse)
            loss_least_val = loss_best_dice_n_mse
            loss_list_val = loss_list_dice_n_mse_val

        # check early stopping conditions
        if index_epoch > min_early_stop:
            loss_least_val, early_stop_patience_counter = early_stopper(loss_list_val, loss_least_val,
                                                                        early_stop_patience_counter, min_early_stop)

            # if early stopping condition meets, break training
            if early_stop_patience_counter > patience_early_stop:
                print(f'Early stopping')
                break


def save_model(args, save_condition, index_epoch, loss, loss_best):
    '''
        Saves the model, best and the latest
    '''

    checkpoint_path = 'checkpoints'
    os.makedirs(checkpoint_path, exist_ok=True)

    # save latest model at each epoch
    save_dict = {'index_epoch': index_epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'optim_state_dict': optimizer.state_dict(),
                 'args': args
                 }

    save_path = os.path.join(checkpoint_path, f'{index_epoch + 1}.pth')
    torch.save(save_dict, save_path)
    save_path = os.path.join(checkpoint_path, f'latest.pth')
    torch.save(save_dict, save_path)

    if loss < loss_best and save_condition:
        loss_best = loss
        save_dict = {'index_epoch': index_epoch + 1,
                     'model_state_dict': model.state_dict(),
                     'optim_state_dict': optimizer.state_dict(),
                     'args': args
                     }

        save_path = os.path.join(checkpoint_path, f'best.pth')
        torch.save(save_dict, save_path)
        print(f'*********************** New best model saved at {args.index_epoch + 1} ***********************')
    return loss_best


def load_model(resume_epoch, model, optimizer):
    '''
        Loads the model
    '''
    load_path = os.path.join('checkpoints', f'{resume_epoch}.pth')

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} doesn\'t exist')

    checkpoint = torch.load(load_path)

    index_epoch = checkpoint['index_epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    args = checkpoint['args']

    print(f'Model successfully loaded from epoch {index_epoch + 1}')

    return args, model, optimizer


fit_model(
    model_name='deep_medic',
    optimizer_name='adam',  # adam, sgd_w_momentum
    learning_rate=0.001,
    num_epochs=10,
    patience_lr_scheduler=3,
    factor_lr_scheduler=0.1,
    patience_early_stop=5,
    min_early_stop=10,
    loss_mode='dice_only',  # dice_only, mse_only, dice_n_mse
    save_condition=True,
    resume_condition=False,
    resume_epoch=1,
    patch_size_normal=25,
    patch_size_low=19,
    patch_size_out=9,
    patch_low_factor=3,
    batch_size=1,
    batch_size_inner=16, # either batch_size or batch_size_inner MUST be set to 1
    train_percentage=0.8
)

if run_mode == 'inference':
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    for index_batch, batch in enumerate(dataloader_inference):
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
