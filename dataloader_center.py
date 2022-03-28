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
import torch.nn.functional as F


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

    def __init__(self, run_mode='train',
                 transform=None,
                 patch_size_normal=25,
                 patch_size_low=19,
                 patch_size_out=9,
                 patch_low_factor=3,
                 label_percentage=0.1,
                 batch_size_inner=100,
                 use_probabilistic=False,
                 create_numpy_dataset=False,
                 dataset_variant='nib',
                 train_percentage=0.8
                 ):

        self.run_mode = run_mode
        self.create_numpy_dataset_cond = create_numpy_dataset
        self.dataset_variant = dataset_variant
        self.patch_size_normal = patch_size_normal
        self.patch_size_low = patch_size_low
        self.patch_size_out = patch_size_out
        self.patch_low_factor = patch_low_factor
        self.batch_size_inner = batch_size_inner
        self.train_percentage = train_percentage
        self.patch_size_low_up = self.patch_size_low * self.patch_low_factor

        self.label_percentage = label_percentage
        self.use_probabilistic = use_probabilistic
        self.fetch_filenames()
        self.create_numpy_dataset()

        self.transform = transform if not transform is None else transforms.Compose([ToTorchTensor()])

    def __getitem__(self, index):
        if self.dataset_variant == 'nib':
            image = self.read_file_nib(self.filenames_image_nib[index])
            label = self.read_file_nib(self.filenames_label_nib[index])
        elif self.dataset_variant == 'npy':

            image = self.read_file_npy(self.filenames_image_npy[index])
            label = self.read_file_npy(self.filenames_label_npy[index])

        if self.run_mode == 'train':
            '''
                # normal = 25
                # low = 19 (57)
                # out = 9
            '''

            # image_temp = np.zeros((512, 512, 512))
            # image_temp[:image.shape[0], :image.shape[1], :image.shape[2]] = image
            # image = image_temp
            #
            # label_temp = np.zeros((512, 512, 512))
            # label_temp[:label.shape[0], :label.shape[1], :label.shape[2]] = label
            # label = label_temp

            if self.batch_size_inner > 1:
                image_patch_normal_stack = torch.zeros((self.batch_size_inner, self.patch_size_normal, self.patch_size_normal, self.patch_size_normal), dtype=torch.float32)
                image_patch_low_stack = torch.zeros((self.batch_size_inner, self.patch_size_low, self.patch_size_low, self.patch_size_low), dtype=torch.float32)
                label_patch_out_stack = torch.zeros((self.batch_size_inner, self.patch_size_out, self.patch_size_out, self.patch_size_out), dtype=torch.int64)

                for index_inner in range(self.batch_size_inner):
                    # extract the three different patches of labels
                    label_patch_normal, label_patch_low_up, label_patch_out = self.get_random_patch(label)

                    # extract the three different patches of images
                    image_patch_normal = self.get_3D_crop(image, self.coordinate_center, self.patch_size_normal)
                    image_patch_low_up = self.get_3D_crop(image, self.coordinate_center, self.patch_size_low_up)
                    image_patch_out = self.get_3D_crop(image, self.coordinate_center, self.patch_size_out)

                    # apply transformation on images
                    image_patch_normal = self.transform(image_patch_normal)
                    image_patch_low_up = self.transform(image_patch_low_up)
                    # image_patch_out = self.transform(image_patch_out)

                    # resize (downsample) the low resolution images and labels
                    image_patch_low = F.avg_pool3d(input=image_patch_low_up.unsqueeze(0), kernel_size=3, stride=None).squeeze(0)

                    # transform the labels to tensors
                    # label_patch_normal = torch.tensor(label_patch_normal, dtype=torch.int64, requires_grad=False)
                    # label_patch_low_up = torch.tensor(label_patch_low_up, dtype=torch.int64, requires_grad=False)
                    label_patch_out = torch.tensor(label_patch_out, dtype=torch.int64, requires_grad=False)

                    image_patch_normal_stack[index_inner] = image_patch_normal.unsqueeze(0)
                    image_patch_low_stack[index_inner] = image_patch_low.unsqueeze(0)
                    label_patch_out_stack[index_inner] = label_patch_out.unsqueeze(0)

                return image_patch_normal_stack.unsqueeze(1), image_patch_low_stack.unsqueeze(1), label_patch_out_stack.unsqueeze(1)
            else:
                # extract the three different patches of labels
                label_patch_normal, label_patch_low_up, label_patch_out = self.get_random_patch(label)

                # extract the three different patches of images
                image_patch_normal = self.get_3D_crop(image, self.coordinate_center, self.patch_size_normal)
                image_patch_low_up = self.get_3D_crop(image, self.coordinate_center, self.patch_size_low_up)
                image_patch_out = self.get_3D_crop(image, self.coordinate_center, self.patch_size_out)

                # apply transformation on images
                image_patch_normal = self.transform(image_patch_normal)
                image_patch_low_up = self.transform(image_patch_low_up)
                # image_patch_out = self.transform(image_patch_out)

                # resize (downsample) the low resolution images and labels
                image_patch_low = F.avg_pool3d(input=image_patch_low_up.unsqueeze(0), kernel_size=3,
                                               stride=None).squeeze(0)

                # transform the labels to tensors
                # label_patch_normal = torch.tensor(label_patch_normal, dtype=torch.int64, requires_grad=False)
                # label_patch_low_up = torch.tensor(label_patch_low_up, dtype=torch.int64, requires_grad=False)
                label_patch_out = torch.tensor(label_patch_out, dtype=torch.int64, requires_grad=False)

                return image_patch_normal.unsqueeze(0), image_patch_low.unsqueeze(0), label_patch_out.unsqueeze(0)

        elif self.run_mode == 'inference':
            # TODO fix uneven dimensions, otherwise run with batch size = 1
            return image, label

    def __len__(self):
        return self.num_samples

    def create_numpy_dataset(self):

        def convert_to_numpy_from_nib(target_dir, filenames):
            os.makedirs(target_dir, exist_ok=True)

            for filename in tqdm(filenames, leave=False):
                data_np = nib.load(filename).get_fdata()

                filename_new = f'{filename[11:-7]}.npy'
                save_path = os.path.join(target_dir, filename_new)
                np.save(save_path, data_np)

        save_dir_train_im = 'imagesTrNP'
        train_filenames_im = self.filenames_image_nib

        save_dir_train_labels = 'labelsTrNP'
        train_filenames_labels = self.filenames_label_nib

        if self.create_numpy_dataset_cond:
            convert_to_numpy_from_nib(target_dir=save_dir_train_im, filenames=train_filenames_im)
            convert_to_numpy_from_nib(target_dir=save_dir_train_labels, filenames=train_filenames_labels)

    def get_label_percentage(self, input, label):
        '''
            Returns the percentage of supplied label in the voxel
        '''
        eps = 1e-9
        denominator = input.shape[0] * input.shape[1] * input.shape[2]
        numerator = np.sum(np.where(input == label, 1, 0))

        return numerator / (denominator + eps)

    def get_rand_index_3D(self, input, height=512, width=512, depth=20, patch_size=57):
        '''
            Returns a random starting index (top-left) of a valid 3D volume
        '''
        patch_size_half = patch_size // 2

        if not self.use_probabilistic:
            #  complete random
            index_h = np.random.randint(patch_size_half, height - patch_size_half)
            index_w = np.random.randint(patch_size_half, width - patch_size_half)
            index_d = np.random.randint(patch_size_half, depth - patch_size_half)
        else:
            # nearby currently selected label
            loop_condition = True
            background_count = 0
            while loop_condition:
                input_cropped = input[patch_size_half:height - patch_size_half, patch_size_half:width - patch_size_half,
                                patch_size_half:depth - patch_size_half]
                indices_all = np.array(np.where(input_cropped == self.current_selected_label))
                # print(indices_all.shape[1])
                if indices_all.shape[1] >= 1:
                    selected_index_w = np.random.randint(indices_all.shape[1])
                    selected_index = indices_all[:, selected_index_w]

                    index_h, index_w, index_d = (selected_index[0]+patch_size_half, selected_index[1]+patch_size_half, selected_index[2]+patch_size_half)
                    # index_h, index_w, index_d = selected_index
                    loop_condition = False
                else:
                    # print('here')
                    if background_count > 0:
                        # if none of the other two labels are present in the image, randomly pick a coordinate
                        index_h = np.random.randint(patch_size_half, height - patch_size_half)
                        index_w = np.random.randint(patch_size_half, width - patch_size_half)
                        index_d = np.random.randint(patch_size_half, depth - patch_size_half)
                        loop_condition = False

                    else:
                        if self.current_selected_label == 1:
                            self.current_selected_label = 2
                            background_count += 1
                        elif self.current_selected_label == 2:
                            self.current_selected_label = 1
                            background_count += 1
                        loop_condition = True

        return (index_h, index_w, index_d)

    def get_3D_crop(self, input, coordinate, patch_size):
        '''
            Returns a 3D patch of an input 3D image given a valid top-left coordinate
        '''
        assert patch_size % 2 == 1, 'Patch size should be an odd number'
        patch_size_half = patch_size // 2

        if len(input.shape) == 3:
            height, width, depth = input.shape

        if depth <= self.patch_size_low * self.patch_low_factor:
            temp_array = np.zeros((height, width, self.patch_size_low * self.patch_low_factor))
            temp_array[:, :, :depth] = input
            input = temp_array
            depth = temp_array.shape[2]

        return input[
               coordinate[0] - patch_size_half: coordinate[0] + patch_size_half + 1,
               coordinate[1] - patch_size_half: coordinate[1] + patch_size_half + 1,
               coordinate[2] - patch_size_half: coordinate[2] + patch_size_half + 1,
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

        if depth <= self.patch_size_low * self.patch_low_factor:
            temp_array = np.zeros((height, width, self.patch_size_low * self.patch_low_factor))
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
            self.coordinate_center = self.get_rand_index_3D(input, height, width, depth, self.patch_size_low_up)

            # if self.coordinate_center[0] + self.patch_size_low_up // 2 >

            patch_normal = self.get_3D_crop(input, self.coordinate_center, self.patch_size_normal)
            patch_low_up = self.get_3D_crop(input, self.coordinate_center, self.patch_size_low_up)
            patch_out = self.get_3D_crop(input, self.coordinate_center, self.patch_size_out)

            # print(f'{patch_normal.shape}\t{patch_low_up.shape}\t{patch_out.shape}')
            loop_condition = False

            # if (patch_normal.shape == (25, 25, 25)) and (patch_low_up.shape == (19, 19, 19)) and (patch_out.shape == (9, 9, 9)):
            #     loop_condition = False

            # if self.use_probabilistic:
            #     # get the percentage of the current label compared to the whole patch
            #     # TODO update this part
            #     pass
            #     self.label_percentage_current = self.get_label_percentage(patch_out, self.current_selected_label)
            #     if self.label_percentage_current > self.label_percentage:
            #         loop_condition = False
            # else:
            #     loop_condition = False

        return patch_normal, patch_low_up, patch_out

    def read_file_nib(self, filename):
        '''
            Reads a nibabel file and returns it in numpy ndarray format
        '''
        try:
            data_nib = nib.load(filename).get_fdata()
        except FileNotFoundError:
            print(f'Error reading file: {filename}')

        return data_nib

    def read_file_npy(self, filename):
        '''
            Reads a npy file and returns it in numpy ndarray format
        '''
        try:
            data_npy = np.load(filename)
        except FileNotFoundError:
            print(f'Error reading file: {filename}')

        return data_npy

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

        num_samples = len(data_meta['training'])

        if self.run_mode == 'train':
            num_samples = int(np.floor(self.train_percentage * num_samples))
        else:
            num_samples = int(np.ceil((1-self.train_percentage) * num_samples))

        self.filenames_image_nib = [current_sample['image'] for current_sample in data_meta['training']][:num_samples]
        self.filenames_label_nib = [current_sample['label'] for current_sample in data_meta['training']][:num_samples]

        self.filenames_image_npy = [os.path.join('ImagesTrNP', f'{filename[11:-7]}.npy') for filename in
                                    self.filenames_image_nib][:num_samples]
        self.filenames_label_npy = [os.path.join('labelsTrNP', f'{filename[11:-7]}.npy') for filename in
                                    self.filenames_label_nib][:num_samples]

        if (not len(self.filenames_image_nib) == len(self.filenames_label_nib)):
            raise Exception('Inconsistent training image/label combination')
        if len(self.filenames_image_nib) == 0:
            raise Exception(f'Error reading {self.run_mode} images')
        if len(self.filenames_label_nib) == 0:
            raise Exception(f'Error reading {self.run_mode} labels')

        elif self.run_mode == 'test':
            # 'TODO' correct the train and test and inference variants
            self.filenames_image_nib = [current_sample for current_sample in data_meta['test']]
            if len(self.filenames_image_nib) == 0:
                raise Exception(f'Error reading {self.run_mode} images')

        self.num_samples = len(self.filenames_image_nib)


