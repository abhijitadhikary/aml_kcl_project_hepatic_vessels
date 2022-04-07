import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import scipy
import torch.nn as nn
import pandas as pd

class GeneralizedDiceLoss(nn.Module):
    '''
        Following the equation from https://arxiv.org/abs/1707.03237 page 3
    '''

    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, im_pred, im_real):
        if len(im_pred.shape) == 4:
            im_pred = im_pred.unsqueeze(0)

        if len(im_real.shape) == 4:
            im_real = im_real.unsqueeze(0)

        im_real = im_real.permute((1, 0, 2, 3, 4))
        im_pred = im_pred.permute((1, 0, 2, 3, 4))

        eps = 1e-12
        sum_1 = torch.sum(im_real[0])
        sum_2 = torch.sum(im_real[1])
        sum_3 = torch.sum(im_real[2])

        weight_1 = 1 / (sum_1 ** 2 + eps) if sum_1 > 0 else 1
        weight_2 = 1 / (sum_2 ** 2 + eps) if sum_2 > 0 else 1
        weight_3 = 1 / (sum_3 ** 2 + eps) if sum_3 > 0 else 1

        numerator_1 = torch.sum(im_real[0] * im_pred[0]) * weight_1
        numerator_2 = torch.sum(im_real[1] * im_pred[1]) * weight_2
        numerator_3 = torch.sum(im_real[2] * im_pred[2]) * weight_3

        numerator = numerator_1 + numerator_2 + numerator_3

        denominator_1 = (torch.sum(im_real[0]) + torch.sum(im_pred[0])) * weight_1
        denominator_2 = (torch.sum(im_real[1]) + torch.sum(im_pred[1])) * weight_2
        denominator_3 = (torch.sum(im_real[2]) + torch.sum(im_pred[2])) * weight_3

        denominator = denominator_1 + denominator_2 + denominator_3

        dice_loss = 1 - ((2 * numerator) / (denominator + eps))

        return dice_loss

# convert predictions to numpy
def prepare_npy_files(path_ensamble):
    name_list = os.listdir(path_ensamble)
    for name in name_list:
        if name == 'Abhijit':
            continue
        for index in tqdm(range(30), leave=False):
            file_path_input = os.path.join(path_ensamble, name, f'{index}.pkl')
            with open(file_path_input, 'rb') as pickle_file:
                try:
                    content = pickle.load(pickle_file)
                    if name == 'Diego':
                        content = content.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
                    elif name == 'Traudi-Beatrice':
                        content = content.cpu().detach().numpy()
                    elif name == 'Kate':
                        content = content.squeeze(0).cpu().detach().numpy()
                    np.save(os.path.join(path_ensamble, name, f'{index}.npy'), content)
                except Exception as e:
                    print(f'({name}) Exception at: {index}: {e}')





if __name__ == '__main__':
    path_ensamble = os.path.join('.', 'Ensamble')
    # prepare_npy_files(path_ensamble)

    # names of all the group members
    names_list = os.listdir(path_ensamble)

    num_inference = 29  # not 30 because the last file of Piyalitt can not be read, thows exception

    # directory to store the average ensambles
    path_avg_ensamble = os.path.join('.', 'Ensamble_avg')
    os.makedirs(path_avg_ensamble, exist_ok=True)

    # define the losses
    criterion_dice = GeneralizedDiceLoss()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    # to hold the Ensamble losses
    # loss_list_dice = []
    # loss_list_mse = []
    # loss_list_dice_n_mse = []

    # df = pd.DataFrame(columns=['dice', 'mse', 'dice_n_mse'])

    loss_names = ['dice', 'mse', 'dice_n_mse']
    num_people = len(names_list)
    print(names_list)
    loss_array_dice = np.zeros((num_inference, num_people+1))
    loss_array_mse = np.zeros((num_inference, num_people+1))
    loss_array_dice_n_mse = np.zeros((num_inference, num_people+1))

    for index_filename in range(num_inference):
        # placeholder to hold the data of each person of the current index
        labels_pred_list = []

        path_labels_true = os.path.join('.', 'labels_true', f'{index_filename}.npy')
        labels_true = np.load(path_labels_true)

        # convert the real labels to one-hot labels
        labels_true_oh = np.zeros((1, 3, labels_true.shape[0], labels_true.shape[1], labels_true.shape[2]))
        labels_true_oh[0, 0, :] = np.where(labels_true==0, 1, 0)
        labels_true_oh[0, 1, :] = np.where(labels_true==1, 1, 0)
        labels_true_oh[0, 2, :] = np.where(labels_true==2, 1, 0)

        depth = labels_true_oh.shape[-1]
        depth_center = (depth // 2)
        depth_new = 16
        depth_new_half = depth_new // 2

        labels_true_oh = labels_true_oh[:, :, :, :, depth_center-depth_new_half:depth_center+depth_new_half]
        labels_true_oh = torch.tensor(labels_true_oh, dtype=torch.float32).squeeze(0)

        print(f'\nFile Index:\t{index_filename}')

        # read the current index file of each person and append to the list
        for index_name, name in enumerate(names_list):
            path_labels_pred = os.path.join(path_ensamble, name, f'{index_filename}.npy')

            labels_pred_proba = np.load(path_labels_pred)

            # getting the center crop of size 16 in the depth dimension to match with other's predictions
            if name=='Abhijit':
                labels_pred_proba = labels_pred_proba[:, :, :, :, depth_center-depth_new_half:depth_center+depth_new_half].squeeze(0)

            labels_pred_proba = torch.softmax(torch.tensor(labels_pred_proba), dim=0).detach().cpu().numpy()
            labels_pred_list.append(labels_pred_proba)

            # convert to torch tensors for loss calculation
            labels_pred_proba = torch.tensor(labels_pred_proba, dtype=torch.float32)

            loss_dice = criterion_dice(labels_pred_proba, labels_true_oh).item()
            loss_mse = criterion_mse(labels_pred_proba, labels_true_oh).item()
            loss_dice_n_mse = loss_dice + loss_mse

            loss_array_dice[index_filename, index_name] = loss_dice
            loss_array_mse[index_filename, index_name] = loss_mse
            loss_array_dice_n_mse[index_filename, index_name] = loss_dice_n_mse

            # print(f'{name}\t\t\tDICE:\t{loss_dice:.5f}\t\tMSE:\t{loss_mse:.5f}\t\tDICE+MSE:\t{loss_dice_n_mse:.5f}')

        # convert to array and calculate the mean
        labels_pred_list = np.array(labels_pred_list)
        labels_pred_mean = np.mean(labels_pred_list, axis=0)

        # loss of the Ensamble
        # convert to torch tensors for loss calculation
        labels_pred_mean = torch.tensor(labels_pred_mean, dtype=torch.float32)

        loss_dice = criterion_dice(labels_pred_mean, labels_true_oh).item()
        loss_mse = criterion_mse(labels_pred_mean, labels_true_oh).item()
        loss_dice_n_mse = loss_dice + loss_mse

        loss_array_dice[index_filename, index_name+1] = loss_dice
        loss_array_mse[index_filename, index_name+1] = loss_mse
        loss_array_dice_n_mse[index_filename, index_name+1] = loss_dice_n_mse
        #
        # loss_list_dice.append(loss_dice)
        # loss_list_mse.append(loss_mse)
        # loss_list_dice_n_mse.append(loss_list_dice_n_mse)

        name = 'Ensamble'
        # print(f'{name}\t\t\tDICE:\t{loss_dice:.5f}\t\tMSE:\t{loss_mse:.5f}\t\tDICE+MSE:\t{loss_dice_n_mse:.5f}')

        # save the average of the ensamble
        path_out_current = os.path.join(path_avg_ensamble, f'{index_filename}.npy')
        np.save(path_out_current, labels_pred_mean)

    df_loss = pd.DataFrame(loss_array_dice_n_mse, columns=names_list.append('Ensamble'))
    name = 'Average Ensamble'
    # print(f'{name}\t\t\tDICE:\t{loss_dice:.5f}\t\tMSE:\t{loss_mse:.5f}\t\tDICE+MSE:\t{loss_dice_n_mse:.5f}')
