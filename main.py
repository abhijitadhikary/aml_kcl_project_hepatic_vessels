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
from datetime import datetime
from imp import reload
import yaml

import json
import sys
import logging

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

batch_size = 1
batch_size_inner = 16
assert batch_size == 1 or batch_size_inner == 1, 'either batch_size or batch_size_inner MUST equal 1 due to size issue'

train_percentage = 0.8


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


# def save_model(args, save_condition, model, optimizer, scheduler, index_epoch, loss, loss_best):
#     '''
#         Saves the model, best and the latest
#     '''
#
#     path_checkpoint = 'checkpoints'
#     os.makedirs(path_checkpoint, exist_ok=True)
#
#     # save latest model at each epoch
#     save_dict = {'index_epoch': index_epoch + 1,
#                  'model_state_dict': model.state_dict(),
#                  'optim_state_dict': optimizer.state_dict(),
#                  'scheduler_state_dict': scheduler.state_dict(),
#                  'args': args
#                  }
#
#     save_path = os.path.join(path_checkpoint, f'{index_epoch + 1}.pth')
#     torch.save(save_dict, save_path)
#     save_path = os.path.join(path_checkpoint, f'latest.pth')
#     torch.save(save_dict, save_path)
#
#     if loss < loss_best and save_condition:
#         loss_best = loss
#         save_dict = {'index_epoch': index_epoch + 1,
#                      'model_state_dict': model.state_dict(),
#                      'optim_state_dict': optimizer.state_dict(),
#                      'args': args
#                      }
#
#         save_path = os.path.join(path_checkpoint, f'best.pth')
#         torch.save(save_dict, save_path)
#         print(f'*********************** New best model saved at {index_epoch + 1} ***********************')
#     return loss_best


def run_inference(
        model_name,
        loss_name,
        patch_size_normal=25,
        patch_size_low=19,
        patch_size_out=9,
        patch_low_factor=3,
        batch_size=1,
        batch_size_inner=16,  # either batch_size or batch_size_inner MUST be set to 1
        train_percentage=0.8,
        load_model=False,
        load_epoch='best'

):
    '''
        Run 3D inference on the validation set. Generates a 3D volume of predicted labels with same shape as the original one
    '''

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    _, _, dataloader_inference = get_dataloaders(
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

    for index_batch, batch in enumerate(dataloader_inference):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        labels_pred, model, criterion_dice, loss_dice, loss_mse = stride_depth_and_inference(
            model=model,
            criterion_dice=criterion_dice,
            criterion_mse=criterion_mse,
            images_real=images,
            labels_real=labels,
            patch_size_normal=patch_size_normal,
            patch_size_low=patch_size_low,
            patch_size_out=patch_size_out,
            patch_low_factor=patch_low_factor
        )

        self.__print__(f'{index_batch}\tLoss DICE:\t{loss_dice}\tLoss MSE:\t{loss_mse}')


class ModelConainer():

    def __init__(self):
        self.__init_model_params__()

    def __init_model_params__(self):
        self.params_model = {
            'model_name': 'deep_medic',
            'patch_size_normal': 25,
            'patch_size_low': 19,
            'patch_size_out': 9,
            'patch_low_factor': 3,
            'run_mode': None,
            'dataset_variant': 'npy',
            'create_numpy_dataset': False,
            'init_timestamp': datetime.now().strftime("%H-%M-%S__%d-%m-%Y")
        }

    def __init_train_params__(self):
        self.params_train = {
            'optimizer_name': 'adam',  # adam, sgd_w_momentum
            'loss_name': 'ce',  # dice, mse, ce, dice_n_mse
            'beta_1': 0.9,
            'beta_2': 0.999,
            'momentum': 0.9,
            'use_amsgrad': True,
            'learning_rate': 0.0001,
            'lr_scheduler_name': 'plateau',
            'patience_lr_scheduler': 3,
            'factor_lr_scheduler': 0.1,
            'early_stop_condition': True,
            'patience_early_stop': 5,
            'early_stop_patience_counter': 0,
            'min_epochs_to_train': 10,
            'num_epochs': 50,
            'min_early_stop': 10,
            'save_every_epoch': True,

            'save_condition': False,  # whether to save the model
            'resume_condition': True,  # whether to resume training

            'resume_dir': '10-58-57__29-03-2022__deep_medic__ce__adam__lr_0.0001__ep_50',
            'resume_epoch': 2,

            'batch_size': 1,
            'batch_size_inner': 16,
            'train_percentage': 0.8,
            'path_checkpoint': 'checkpoints',
            'path_checkpoint_full': '',
            'dirname_checkpoint': '',
            'filename_params': 'params.json',
            'filename_logger': 'logger.txt',
            'path_params_full': '',
            'path_logger_full': ''
        }

        self.loss_dict_train = {
            'total': [],
            'dice': [],
            'mse': [],
            'ce': [],
            'dice_n_mse': []
        }

        self.loss_dict_val = {
            'total': [],
            'dice': [],
            'mse': [],
            'ce': [],
            'dice_n_mse': []
        }

        self.loss_best_train = {
            'total': np.inf,
            'dice': np.inf,
            'mse': np.inf,
            'ce': np.inf,
            'dice_n_mse': np.inf
        }

        self.loss_best_val = {
            'total': np.inf,
            'dice': np.inf,
            'mse': np.inf,
            'ce': np.inf,
            'dice_n_mse': np.inf
        }

        assert self.params_train['batch_size'] == 1 or self.params_train[
            'batch_size_inner'] == 1, 'either must be 1, size issue'

    def __init_inference_params__(self):
        self.params_inference = {
            'loss_name': 'dice',
            'batch_size': 1,
            'batch_size_inner': 16,  # either batch_size or batch_size_inner MUST be set to 1
            'train_percentage': 0.8,
            'load_model': False,
            'load_epoch': 'best'
        }

    def __setup_logger__(self):

        if not self.params_train['resume_condition']:
            reload(logging)

        logging.basicConfig(filename=self.params_train['path_logger_full'], encoding='utf-8', level=logging.DEBUG)

    def __create_params_file__(self):
        params_dict = {
            'params_model': self.params_model,
            'params_train': self.params_train
        }

        params_dict = json.dumps(params_dict, indent=4, sort_keys=False)

        with open(self.params_train['path_params_full'], 'w') as outfile:
            outfile.write(params_dict)

        self.__print__(f'{"*" * 100}')
        self.__print__('\t\tTraining starting with params:')
        self.__print__(f'{"*" * 100}')
        self.__print__(f'{params_dict}')
        self.__print__(f'{"*" * 100}')

    def __create_checkpoint_dir__(self):
        if self.params_train['resume_condition']:
            self.params_train['dirname_checkpoint'] = self.params_train['resume_dir'][:11]
            self.params_train['path_checkpoint_full'] = self.params_train['resume_dir']
        else:
            self.params_train['dirname_checkpoint'] = f'{self.params_model["init_timestamp"]}__' \
                                                      f'{self.params_model["model_name"]}__' \
                                                      f'{self.params_train["loss_name"]}__' \
                                                      f'{self.params_train["optimizer_name"]}__' \
                                                      f'lr_{self.params_train["learning_rate"]}__' \
                                                      f'ep_{self.params_train["num_epochs"]}'

            self.params_train['path_checkpoint_full'] = os.path.join(self.params_train['path_checkpoint'],
                                                                     self.params_train['dirname_checkpoint'])

        self.params_train['path_params_full'] = os.path.join(self.params_train['path_checkpoint_full'],
                                                             self.params_train['filename_params'])
        self.params_train['path_logger_full'] = os.path.join(self.params_train['path_checkpoint_full'],
                                                             self.params_train['filename_logger'])
        os.makedirs(self.params_train['path_checkpoint_full'], exist_ok=True)

    def train(self):
        self.__init_train_params__()
        self.__create_checkpoint_dir__()
        self.__create_params_file__()
        self.__setup_logger__()
        self.__fit_model__()

    def inference(self):
        self.__init_inference_params__()
        self.__run_inference__()

    def __run_inference__(self):
        pass

    def __set_device__(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __get_dataloaders__(self, run_mode):
        if run_mode == 'train':
            self.dataset_train = DatasetHepatic(
                run_mode='train',
                label_percentage=0.0001,
                use_probabilistic=True,
                patch_size_normal=self.params_model['patch_size_normal'],
                patch_size_low=self.params_model['patch_size_low'],
                patch_size_out=self.params_model['patch_size_out'],
                patch_low_factor=self.params_model['patch_low_factor'],
                create_numpy_dataset=self.params_model['create_numpy_dataset'],
                dataset_variant=self.params_model['dataset_variant'],
                batch_size_inner=self.params_train['batch_size_inner'],
                train_percentage=self.params_train['train_percentage']
            )

            self.dataset_val = DatasetHepatic(
                run_mode='val',
                label_percentage=0.0001,
                use_probabilistic=True,
                patch_size_normal=self.params_model['patch_size_normal'],
                patch_size_low=self.params_model['patch_size_low'],
                patch_size_out=self.params_model['patch_size_out'],
                patch_low_factor=self.params_model['patch_low_factor'],
                create_numpy_dataset=self.params_model['create_numpy_dataset'],
                dataset_variant=self.params_model['dataset_variant'],
                batch_size_inner=self.params_train['batch_size_inner'],
                train_percentage=self.params_train['train_percentage']
            )

            self.dataloader_train = DataLoader(
                self.dataset_train,
                batch_size=self.params_train['batch_size'],
                shuffle=True
            )

            self.dataloader_val = DataLoader(
                self.dataset_val,
                batch_size=self.params_train['batch_size'],
                shuffle=True
            )

        elif run_mode == 'inference':
            self.dataset_inference = DatasetHepatic(
                run_mode='inference',
                label_percentage=0.0001,
                use_probabilistic=True,
                patch_size_normal=self.params_model['patch_size_normal'],
                patch_size_low=self.params_model['patch_size_low'],
                patch_size_out=self.params_model['patch_size_out'],
                patch_low_factor=self.params_model['patch_low_factor'],
                create_numpy_dataset=self.params_model['create_numpy_dataset'],
                dataset_variant=self.params_model['dataset_variant']
            )

            self.dataloader_inference = DataLoader(
                self.dataset_inference,
                batch_size=self.params_inference['batch_size'],
                shuffle=True
            )

    def __define_model__(self):
        if self.params_model['model_name'] == 'deep_medic':
            self.model = DeepMedic().to(self.device)

    def __define_criterions__(self):
        self.criterion_mse = nn.MSELoss()
        self.criterion_dice = GeneralizedDiceLoss().dice
        self.criterion_ce = nn.CrossEntropyLoss()

    def __define_optimizr__(self):
        if self.params_train['optimizer_name'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.params_train['learning_rate'],
                betas=(self.params_train['beta_1'], self.params_train['beta_2']),
                amsgrad=self.params_train['use_amsgrad']
            )

        elif self.params_train['optimizer_name'] == 'sgd_w_momentum':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.params_train['learning_rate'],
                momentum=self.params_train['momentum']
            )
        else:
            raise NotImplementedError(f'Invalid choice of optimizer:\t{self.params_train["optimizer_name"]}')

    def __define_lr_scheduler__(self):
        if self.params_train['lr_scheduler_name'] == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.params_train['patience_lr_scheduler'],
                factor=self.params_train['factor_lr_scheduler'],
                verbose=True)

    def __put_to_device__(self, device, tensors):
        for index, tensor in enumerate(tensors):
            tensors[index] = tensor.to(device)
        return tensors

    def __get_one_hot_labels__(self, input, labels, squeeze_dim=None):
        output = torch.zeros((input.shape[0], len(labels), input.shape[2], input.shape[3], input.shape[4]),
                             dtype=input.dtype).to(input.device)

        if not squeeze_dim is None:
            for index, label in enumerate(labels):
                output[:, index] = torch.where(input == label, 1, 0).squeeze(squeeze_dim)
        else:
            for index, label in enumerate(labels):
                output[:, label] = torch.where(input == label, 1, 0)

        return output

    def __save_model__(self):
        '''
            Saves the model, best and the latest
        '''
        if self.params_train['save_condition']:

            save_dict = {
                'index_epoch': self.index_epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                'params_model': self.params_model,
                'params_train': self.params_train,
                'loss_dict_train': self.loss_dict_train,
                'loss_dict_val': self.loss_dict_val,
                'loss_best_train': self.loss_best_train,
                'loss_best_val': self.loss_best_val
            }

            # save models at each epoch
            if self.params_train['save_every_epoch']:
                save_path = os.path.join(self.params_train['path_checkpoint_full'], f'{self.index_epoch + 1}.pth')
                torch.save(save_dict, save_path)

            # save the latest model
            save_path = os.path.join(self.params_train['path_checkpoint_full'], f'latest.pth')
            torch.save(save_dict, save_path)

            if self.found_best_loss_flag:
                save_path = os.path.join(self.params_train['path_checkpoint_full'], f'best.pth')
                torch.save(save_dict, save_path)
                self.__print__(f'{"*" * 10}\tNew best model saved at:\t{self.index_epoch + 1}\t{"*" * 10}')

    def __load_model__(self):
        '''
            Loads the model
        '''
        if self.params_train['resume_condition']:
            filename_checkpoint = f'{self.params_train["resume_epoch"]}.pth'
            load_path = os.path.join(self.params_train['path_checkpoint'],
                                     self.params_train['resume_dir'],
                                     filename_checkpoint)

            if not os.path.exists(load_path):
                raise FileNotFoundError(f'File {load_path} doesn\'t exist')

            checkpoint = torch.load(load_path)

            self.index_epoch = checkpoint['index_epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            self.params_model = checkpoint['params_model']
            self.params_train = checkpoint['params_train']
            self.loss_dict_train = checkpoint['loss_dict_train']
            self.loss_dict_val = checkpoint['loss_dict_val']
            self.loss_best_train = checkpoint['loss_best_train']
            self.loss_best_val = checkpoint['loss_best_val']

            self.__print__(f'Model loaded from epoch:\t{self.index_epoch}')
            self.index_epoch += 1
            self.start_epoch = self.index_epoch
            self.__print__(f'Resuming training from epoch:\t{self.index_epoch}')

    def __early_stop__(self):
        '''
            If early stopping condition meets, break training
        '''
        if self.index_epoch > self.params_train['min_epochs_to_train']:
            if self.params_train['early_stop_patience_counter'] > self.params_train['patience_early_stop']:
                self.__print__(f'Early stopping at epoch:\t{self.index_epoch + 1}')

    def __run_epoch__(self, dataloader, run_mode):
        '''
            Runs one epoch of training and validation loops
        '''
        loss_list_total = []
        loss_list_dice = []
        loss_list_ce = []
        loss_list_mse = []
        loss_list_dice_n_mse = []
        loss_total, loss_dice, loss_mse, loss_ce, loss_dice_n_mse = np.Inf, np.Inf, np.Inf, np.Inf, np.Inf

        for index_batch, batch in enumerate(dataloader):
            if run_mode == 'train':
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()

            (image_patch_normal, image_patch_low, label_patch_out_real) = self.__put_to_device__(self.device, batch)

            if self.params_train['batch_size_inner'] > 1:
                image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.squeeze(
                    0), image_patch_low.squeeze(0), label_patch_out_real.squeeze(0)

            # forward pass
            label_patch_out_pred = self.model.forward((image_patch_normal, image_patch_low))

            # convert label_patch_out_real to one hot
            label_patch_out_real_one_hot = self.__get_one_hot_labels__(label_patch_out_real, labels=[0, 1, 2],
                                                                       squeeze_dim=1)

            # generalized dice loss_total # dice, mse, dice_n_mse
            if self.params_train['loss_name'] == 'dice':
                loss_dice = self.criterion_dice(F.softmax(label_patch_out_pred.float(), dim=1),
                                                label_patch_out_real_one_hot.float())
                loss_list_dice.append(loss_dice.item())
                loss_total = loss_dice
            elif self.params_train['loss_name'] == 'mse':
                loss_mse = self.criterion_mse(F.softmax(label_patch_out_pred.float(), dim=1),
                                              label_patch_out_real_one_hot.float())
                loss_list_mse.append(loss_mse.item())
                loss_total = loss_mse
            elif self.params_train['loss_name'] == 'ce':
                loss_ce = self.criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1).long())
                loss_list_ce.append(loss_ce.item())
                loss_total = loss_ce
            elif self.params_train['loss_name'] == 'dice_n_mse':
                loss_dice = self.criterion_dice(F.softmax(label_patch_out_pred.float(), dim=1),
                                                label_patch_out_real_one_hot.float())
                loss_mse = self.criterion_mse(F.softmax(label_patch_out_pred.float(), dim=1),
                                              label_patch_out_real_one_hot.float())
                loss_dice_n_mse = loss_dice + loss_mse
                loss_total = loss_dice_n_mse
                loss_list_dice.append(loss_dice.item())
                loss_list_mse.append(loss_mse.item())
                loss_list_dice_n_mse.append(loss_dice_n_mse.item())
            else:
                raise NotImplementedError(f'Invalid criterion selected:\t{self.params_train["loss_name"]}')

            loss_list_total.append(loss_total)

            if run_mode == 'train':
                # calculate gradients and update weights
                loss_total.backward()
                self.optimizer.step()
            sep = '\t' if run_mode == 'train' else '\t\t'
            self.__print__(f'\tBatch:\t[{index_batch + 1} / {len(dataloader)}]'
                           f'\n\t\t{str(run_mode).upper()}{sep}-->\t\tLoss ({self.params_train["loss_name"]}):\t\t{loss_total.item():.5f}')

            logging.debug('Debugger working here')
            # ###############################
            # loss_mse = self.criterion_mse(F.softmax(label_patch_out_pred.float(), dim=1),
            #                               label_patch_out_real_one_hot.float())
            # loss_list_mse.append(loss_mse.item())
            #
            # print(f'Loss ({self.params_train["loss_name"]}):\t{loss_total.item():.5f}\t\tMSE:\t{loss_mse.item()}')
            # ###############################

            # print(loss_dice.item())
            # print(loss_mse.item())
            # print(loss_list_dice_n_mse.item())
            break

        # loss_dice = sum(loss_list_dice) / len(loss_list_dice)
        # loss_mse = 0
        # loss_dice_n_mse = 0

        if len(loss_list_total) > 0:
            loss_total = sum(loss_list_total) / len(loss_list_total)
        if len(loss_list_dice) > 0:
            loss_dice = sum(loss_list_dice) / len(loss_list_dice)
        if len(loss_list_mse) > 0:
            loss_mse = sum(loss_list_mse) / len(loss_list_mse)
        if len(loss_list_ce) > 0:
            loss_ce = sum(loss_list_ce) / len(loss_list_ce)
        if len(loss_list_dice_n_mse) > 0:
            loss_dice_n_mse = sum(loss_list_dice_n_mse) / len(loss_list_dice_n_mse)

        if run_mode == 'train':
            self.loss_dict_train['total'].append(loss_total.item())
            self.loss_dict_train['dice'].append(loss_dice)
            self.loss_dict_train['mse'].append(loss_mse)
            self.loss_dict_train['ce'].append(loss_ce)
            self.loss_dict_train['dice_n_mse'].append(loss_mse)

        elif run_mode == 'val':
            self.loss_dict_val['total'].append(loss_total.item())
            self.loss_dict_val['dice'].append(loss_dice)
            self.loss_dict_val['mse'].append(loss_mse)
            self.loss_dict_val['ce'].append(loss_ce)
            self.loss_dict_val['dice_n_mse'].append(loss_mse)

    def __update_best_losses__(self):
        '''
            Updates the best loss found so far
        '''
        self.found_best_loss_flag = False

        for (key, value) in self.loss_dict_train.items():
            if self.loss_dict_train[key][-1] < min(self.loss_dict_train[key]):
                self.loss_best_train[key] = self.loss_dict_train[key][-1]
                # if self.params_train['loss_name'] == key:
                #     self.found_best_loss_flag = True

        for (key, value) in self.loss_dict_val.items():
            if self.loss_dict_val[key][-1] < min(self.loss_dict_val[key]):
                self.loss_best_val[key] = self.loss_dict_val[key][-1]
                if self.params_train['loss_name'] == key:
                    self.found_best_loss_flag = True

    def __print__(self, message):
        print(message)
        logging.debug(message)
        logging.debug('working')

    def __fit_model__(self):
        '''
            Trains and validates a model given hyperparameters, model and optimizer name, dataloaders, num_epochs
        '''

        # set up dataloaders, model, criterions, optimizers, schedulers
        self.__set_device__()
        self.__get_dataloaders__('train')
        self.__define_model__()
        self.__define_criterions__()
        self.__define_optimizr__()
        self.__define_lr_scheduler__()

        # variables to keep track of training progress
        self.start_epoch = 0
        self.break_training_condition = False
        self.end_epoch = self.params_train['num_epochs']

        self.__load_model__()

        for index_epoch in range(self.start_epoch, self.end_epoch):
            time_start = time.time()
            self.index_epoch = index_epoch

            # train
            self.__run_epoch__(
                dataloader=self.dataloader_train,
                run_mode='train'
            )

            # validation
            with torch.no_grad():
                self.__run_epoch__(
                    dataloader=self.dataloader_val,
                    run_mode='val'
                )

            # choose which loss to put into the scheduler
            self.lr_scheduler.step(self.loss_dict_val['total'][-1])

            duration = time.time() - time_start

            self.__print__(f'\n{"-" * 100}'
                           f'\nEpoch:\t[{index_epoch + 1} / {self.end_epoch}]\t\t'
                           f'Time:\t{duration:.2f} s'
                           f'\n\tTRAIN\t\t-->\t\tLoss Total:\t\t{self.loss_dict_train["total"][-1]:.5f}'
                           f'\n\tVAL\t\t\t-->\t\tLoss Total:\t\t{self.loss_dict_val["total"][-1]:.5f}'
                           f'\n{"-" * 100}\n')

            self.__update_best_losses__()
            self.__save_model__()
            self.__early_stop__()

            if self.break_training_condition:
                break


model_container = ModelConainer()
model_container.train()
print()
# fit_model(
#     model_name='deep_medic',
#     optimizer_name='adam',  # adam, sgd_w_momentum
#     learning_rate=0.0001,
#     num_epochs=50,
#     patience_lr_scheduler=3,
#     factor_lr_scheduler=0.1,
#     patience_early_stop=5,
#     min_early_stop=10,
#     loss_name='dice',  # dice, mse, dice_n_mse
#     save_condition=False,
#     resume_condition=True,
#     resume_epoch=3,
#
#     patch_size_normal=25,
#     patch_size_low=19,
#     patch_size_out=9,
#     patch_low_factor=3,
#     batch_size=1,
#     batch_size_inner=16, # either batch_size or batch_size_inner MUST be set to 1
#     train_percentage=0.8
# )

# run_inference(
#         model_name='deep_medic',
#         loss_name='dice',
#         patch_size_normal=25,
#         patch_size_low=19,
#         patch_size_out=9,
#         patch_low_factor=3,
#         batch_size=1,
#         batch_size_inner=16,  # either batch_size or batch_size_inner MUST be set to 1
#         train_percentage=0.8,
#         load_model=False,
#         load_epoch='best'
# )
