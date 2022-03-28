import copy
import json
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# % matplotlib inline
import nibabel as nib
from tqdm import tqdm
import json

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms



class ToTorchTensor():
    '''
        Transforms a numpy ndarray to a torch tensor of the supplied datatype
    '''
    def __call__(self, input, datatype=torch.float32, requires_grad=True):
        return torch.tensor(input, dtype=datatype, requires_grad=requires_grad)

class DatasetHepatic(Dataset):
    '''
        min = 24
        max = 181
        median = 49
        mean = 69
    '''
    def __init__(self, run_mode='train', transform=None, patch_size_normal=25, patch_size_low=19, patch_size_out=9, patch_low_factor=3, label_percentage=0.1, use_probabilistic=False):
    # def __init__(self, run_mode='train', transform=None, patch_size_normal=510, patch_size_low=510//3, patch_size_out=500//3, patch_low_factor=3, label_percentage=0.1, use_probabilistic=False):

        self.run_mode = run_mode
        self.patch_size_normal = patch_size_normal
        self.patch_size_low = patch_size_low
        self.patch_size_out = patch_size_out
        self.patch_low_factor = patch_low_factor
        self.label_percentage = label_percentage
        self.use_probabilistic = use_probabilistic
        self.fetch_filenames()

        self.transform = transform if not transform is None else transforms.Compose([ToTorchTensor()])

    def __getitem__(self, index):

        image = self.read_file_nib(self.filenames_image[index])
        label = self.read_file_nib(self.filenames_label[index])

        if self.run_mode == 'train':
            '''
                # normal = 25
                # low = 19 (57)
                # out = 9
            '''

            label_patch_normal, label_patch_low, label_patch_out = self.get_random_patch(label)
            image_patch_normal = self.get_3D_crop(image, self.coordinate_center, self.patch_size_normal)
            image_patch_low = self.get_3D_crop(image, self.coordinate_center, self.patch_size_low)
            image_patch_out = self.get_3D_crop(image, self.coordinate_center, self.patch_size_out)

            image_patch_normal = self.transform(image_patch_normal)
            image_patch_low = self.transform(image_patch_low)
            image_patch_out = self.transform(image_patch_out)

            label_patch_normal = torch.tensor(label_patch_normal, dtype=torch.int64, requires_grad=False)
            label_patch_low = torch.tensor(label_patch_low, dtype=torch.int64, requires_grad=False)
            label_patch_out = torch.tensor(label_patch_out, dtype=torch.int64, requires_grad=False)

            return image_patch_normal.unsqueeze(0), image_patch_low.unsqueeze(0), label_patch_out.unsqueeze(0)

        elif self.run_mode == 'inference':

            return image, label





    def __len__(self):
        return self.num_samples

    def get_label_percentage(self, input, label):
        '''
            Returns the percentage of supplied label in the voxel
        '''
        eps = 1e-9
        denominator = input.shape[0] * input.shape[1] * input.shape[2]
        numerator = np.sum(np.where(input == label, 1, 0))

        return numerator / (denominator + eps)

    def get_rand_index_3D(self, height=512, width=512, depth=20, patch_size=57):
        '''
            Returns a random starting index (top-left) of a valid 3D volume
        '''
        patch_size_half = patch_size//2
        index_h = np.random.randint(patch_size_half, height-patch_size_half)
        index_w = np.random.randint(patch_size_half, width-patch_size_half)
        index_d = np.random.randint(patch_size_half, depth-patch_size_half)

        return (index_h, index_w, index_d)

    def get_3D_crop(self, input, coordinate, patch_size):
        '''
            Returns a 3D patch of an input 3D image given a valid top-left coordinate
        '''
        assert patch_size % 2 == 1, 'Patch size should be an odd number'
        patch_size_half = patch_size // 2

        if len(input.shape) == 3:
            height, width, depth = input.shape

        if depth <= self.patch_size_low*self.patch_low_factor:
            temp_array = np.zeros((height, width, self.patch_size_low*self.patch_low_factor))
            temp_array[:, :, :depth] = input
            input = temp_array
            depth = temp_array.shape[2]

        return input[
               coordinate[0]-patch_size_half: coordinate[0]+patch_size_half+1,
               coordinate[1]-patch_size_half: coordinate[1]+patch_size_half+1,
               coordinate[2]-patch_size_half: coordinate[2]+patch_size_half+1,
               ]

    def set_probabilistic_label(self):
        '''
            Randomly with equal probability select one of the three labels to be the current label
        '''
        label_probability = np.random.rand()

        if label_probability > 0.5:
            self.current_selected_label = 1
        else:
            self.current_selected_label = 2

        # if label_probability > 0.66:
        #     self.current_selected_label = 2
        # elif label_probability < 0.33:
        #     self.current_selected_label = 1
        # else:
        #     self.current_selected_label = 0

    def get_random_patch(self, input):
        '''
            Returns a valid cubic sub-volume with edge lenth = patch_size from a supplied 3D input volume image_input
        '''
        # a = copy.deepcopy(input)
        if len(input.shape) == 3:
            height, width, depth = input.shape

        if depth <= self.patch_size_low*self.patch_low_factor:
            temp_array = np.zeros((height, width, self.patch_size_low*self.patch_low_factor))
            temp_array[:, :, :depth] = input
            input = temp_array
            depth = temp_array.shape[2]

        # temp_array = np.zeros((height, width, 512))
        # temp_array[:, :, :depth] = input
        # input = temp_array
        # depth = temp_array.shape[2]

        loop_condition = True
        if self.use_probabilistic:
            self.set_probabilistic_label()

        # keep sampling a new patch until the current label meets the desired overall percentage
        while loop_condition:
            # get a valid coordinate and extract the patch
            self.coordinate_center = self.get_rand_index_3D(height, width, depth, self.patch_size_low*self.patch_low_factor)

            patch_normal = self.get_3D_crop(input, self.coordinate_center, self.patch_size_normal)
            patch_low = self.get_3D_crop(input, self.coordinate_center, self.patch_size_low)
            patch_out = self.get_3D_crop(input, self.coordinate_center, self.patch_size_out)

            # print(f'{patch_normal.shape}\t{patch_low.shape}\t{patch_out.shape}')
            # loop_condition = False

            # if (patch_normal.shape == (25, 25, 25)) and (patch_low.shape == (19, 19, 19)) and (patch_out.shape == (9, 9, 9)):
            #     loop_condition = False

            if self.use_probabilistic:
                # get the percentage of the current label compared to the whole patch
                # TODO update this part
                pass
                self.label_percentage_current = self.get_label_percentage(patch_out, self.current_selected_label)
                if self.label_percentage_current > self.label_percentage:
                    loop_condition = False
            else:
                loop_condition = False

        return patch_normal, patch_low, patch_out

    def read_file_nib(self, filename):
        '''
            Reads a nibabel file and returns it in numpy ndarray format
        '''
        try:
            data_nib = nib.load(filename).get_fdata()
        except FileNotFoundError:
            print(f'Error reading file: {filename}')

        return data_nib

    def fetch_filenames(self, path_meta='dataset.json'):
        '''
            Reads the dataset.json file and extracts the training and test image and/or labels
        :return:
        '''
        try:
            with open(path_meta) as file_meta:
                data_meta = json.loads(file_meta.read())
        except FileNotFoundError:
            print(f'Meta file: {self.path_meta} not found')

        if self.run_mode in ['train', 'inference']:
            self.filenames_image = [current_sample['image'] for current_sample in data_meta['training']]
            self.filenames_label = [current_sample['label'] for current_sample in data_meta['training']]

            if (not len(self.filenames_image) == len(self.filenames_label)):
                raise Exception('Inconsistent training image/label combination')
            if len(self.filenames_image) == 0:
                raise Exception(f'Error reading {self.run_mode} images')
            if len(self.filenames_label) == 0:
                raise Exception(f'Error reading {self.run_mode} labels')

        elif self.run_mode == 'test':
            # 'TODO' correct the train and test and inference variants
            self.filenames_image = [current_sample for current_sample in data_meta['test']]
            if len(self.filenames_image) == 0:
                raise Exception(f'Error reading {self.run_mode} images')

        self.num_samples = len(self.filenames_image)




