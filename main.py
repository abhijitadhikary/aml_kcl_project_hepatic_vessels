import os
import time
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
import nibabel as nib
from tqdm import tqdm
import json
import cv2

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# from dataloader_top_left import DatasetHepatic
from dataloader_center import DatasetHepatic

# from network_resnet_deepmedic import DeepMedic
from network_deepmedic import DeepMedic

dataset = DatasetHepatic(run_mode='train', label_percentage=-1, use_probabilistic=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# for i in range(20):
#     batch = next(iter(dataloader))
#     print(f'{i}\t{batch[0].shape}\t{batch[1].shape}\t{batch[2].shape}')

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

learning_rate = 0.0001
momentum = 0.9
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = DeepMedic().to(device)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), amsgrad=True)
criterion_mse = nn.MSELoss()
# criterion_mse = get_mse_loss
criterion_dice = GeneralizedDiceLoss().dice
criterion_ce = nn.CrossEntropyLoss()
num_epochs = 10
run_mode = 'train'

for index_epoch in tqdm(range(num_epochs), leave=False):
    time_start = time.time()
    loss_list_mse = []
    for index_batch, batch in enumerate(dataloader):
        if run_mode == 'train':
            model.train()
            optimizer.zero_grad()
            image_patch_normal, image_patch_low, label_patch_out_real = batch
            image_patch_normal, image_patch_low, label_patch_out_real = image_patch_normal.to(device), image_patch_low.to(device), label_patch_out_real.to(device)

            label_patch_out_pred_logits = model.forward((image_patch_normal, image_patch_low))

            label_patch_out_pred = label_patch_out_pred_logits
            # print(label_patch_out_pred)

            # cross-entropy loss
            # loss = criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1))




            # convert label_patch_out_real to one hot
            label_patch_out_real_one_hot = torch.zeros_like(label_patch_out_pred).to(device)
            label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 0, 1)
            label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 1)
            label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 2, 1)

            # generalized dice loss
            loss = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)

            loss.backward()
            optimizer.step()

            loss_list_mse.append(loss.item())

            print(loss.item())

    loss_mse_epoch = sum(loss_list_mse) / len(loss_list_mse)

    duration = time.time() - time_start
    print(f'Epoch:\t[{index_epoch+1} / {num_epochs}]\t\tTime:\t{duration} s\t\tMSE:\t{loss_mse_epoch:.5f}')

print('Done')


# def generalized_dice_loss(im_real, im_pred):
#     total_voxels = im_real.shape[0] * im_real.shape[1] * im_real.shape[2], im_real.shape[3]
#
#     weights = [1 / np.power(np.sum(np.where(im_real == index, 1, 0)), 2) for index in range(3)]



