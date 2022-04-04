import os
import time
import numpy as np
import nibabel as nib
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
# %matplotlib inline
import copy
from datetime import datetime
from imp import reload
import json
import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms


# DATALOADER
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# for reproducability
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ToTorchTensor:
    '''
        Transforms a numpy ndarray to a torch tensor of the supplied datatype
    '''

    def __call__(self, input, datatype=torch.float32, requires_grad=False):
        return torch.tensor(input, dtype=datatype, requires_grad=requires_grad)


class ZeroOneNormalize:
    '''
        Transforms a numpy ndarray to a torch tensor of the supplied datatype
    '''

    def __call__(self, input):
        input = input - torch.min(input)
        input = input / torch.max(input)
        return input


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

        if transform is None:
            self.transform = transforms.Compose([
                ToTorchTensor(),
                transforms.Normalize(mean=0.5, std=0.5)
            ])
        else:
            self.transform = transform
        # self.transform = transform if not transform is None else transforms.Compose([ToTorchTensor()])

    def __getitem__(self, index):
        if self.dataset_variant == 'nib':
            image = self.read_file_nib(self.filenames_image_nib[index])
            label = self.read_file_nib(self.filenames_label_nib[index])
        elif self.dataset_variant == 'npy':
            image = self.read_file_npy(self.filenames_image_npy[index])
            label = self.read_file_npy(self.filenames_label_npy[index])

        # index of the original filenames as in the dataset folders
        index_filename = self.filenames_image_npy[index][25:28]

        if self.run_mode in ['train', 'val']:
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
                image_patch_normal_stack = torch.zeros(
                    (self.batch_size_inner, self.patch_size_normal, self.patch_size_normal, self.patch_size_normal),
                    dtype=torch.float32)
                image_patch_low_up_stack = torch.zeros(
                    (self.batch_size_inner, self.patch_size_low_up, self.patch_size_low_up, self.patch_size_low_up),
                    dtype=torch.float32)
                label_patch_out_stack = torch.zeros(
                    (self.batch_size_inner, self.patch_size_out, self.patch_size_out, self.patch_size_out),
                    dtype=torch.int64)

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

                    # # resize (downsample) the low resolution images and labels
                    # image_patch_low = F.avg_pool3d(input=image_patch_low_up.unsqueeze(0), kernel_size=3, stride=None).squeeze(0)

                    # transform the labels to tensors
                    # label_patch_normal = torch.tensor(label_patch_normal, dtype=torch.int64)
                    # label_patch_low_up = torch.tensor(label_patch_low_up, dtype=torch.int64)
                    label_patch_out = torch.tensor(label_patch_out, dtype=torch.int64)

                    image_patch_normal_stack[index_inner] = image_patch_normal.unsqueeze(0)
                    image_patch_low_up_stack[index_inner] = image_patch_low_up.unsqueeze(0)
                    label_patch_out_stack[index_inner] = label_patch_out.unsqueeze(0)

                return image_patch_normal_stack.unsqueeze(1), image_patch_low_up_stack.unsqueeze(
                    1), label_patch_out_stack.unsqueeze(1)
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
                # image_patch_low = F.avg_pool3d(input=image_patch_low_up.unsqueeze(0), kernel_size=3, stride=None).squeeze(0)

                # transform the labels to tensors
                # label_patch_normal = torch.tensor(label_patch_normal, dtype=torch.int64)
                # label_patch_low_up = torch.tensor(label_patch_low_up, dtype=torch.int64)
                label_patch_out = torch.tensor(label_patch_out, dtype=torch.int64)

                return image_patch_normal.unsqueeze(0), image_patch_low_up.unsqueeze(0), label_patch_out.unsqueeze(0)

        elif self.run_mode == 'inference':
            # TODO fix uneven dimensions, otherwise run with batch size = 1
            image = self.transform(image)
            return image, label, index_filename

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

        if self.use_probabilistic:
            # nearby currently selected label
            loop_condition = True
            background_count = 0
            while loop_condition:
                # crop the image so that the patch does not go outside the area of the image
                input_cropped = input[patch_size_half:height - patch_size_half, patch_size_half:width - patch_size_half,
                                patch_size_half:depth - patch_size_half]
                # all indices of the cropped image equal to the current selected label category
                indices_all = np.array(np.where(input_cropped == self.current_selected_label))
                # print(indices_all.shape[1])
                if indices_all.shape[1] >= 1:
                    selected_index_w = np.random.randint(indices_all.shape[1])
                    selected_index = indices_all[:, selected_index_w]

                    index_h, index_w, index_d = (
                        selected_index[0] + patch_size_half, selected_index[1] + patch_size_half,
                        selected_index[2] + patch_size_half)
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
        else:
            #  complete random
            index_h = np.random.randint(patch_size_half, height - patch_size_half)
            index_w = np.random.randint(patch_size_half, width - patch_size_half)
            index_d = np.random.randint(patch_size_half, depth - patch_size_half)

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
        mode = 'major'  # equal, biased
        if mode == 'biased':
            if label_probability > 0.5:
                self.current_selected_label = 1
            else:
                self.current_selected_label = 2
        elif mode == 'equal':
            if label_probability > 0.66:
                self.current_selected_label = 2
            elif label_probability < 0.33:
                self.current_selected_label = 1
            else:
                self.current_selected_label = 0
        elif mode == 'major':
            if label_probability > 0.45:
                self.current_selected_label = 2
            elif label_probability < 0.45:
                self.current_selected_label = 1
            else:
                self.current_selected_label = 0

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

            self.filenames_image_nib = [current_sample['image'] for current_sample in data_meta['training']][
                                       :num_samples]
            self.filenames_label_nib = [current_sample['label'] for current_sample in data_meta['training']][
                                       :num_samples]

            self.filenames_image_npy = [os.path.join('.', 'imagesTrNP', f'{filename[11:-7]}.npy') for filename in
                                        self.filenames_image_nib]
            self.filenames_label_npy = [os.path.join('.', 'labelsTrNP', f'{filename[11:-7]}.npy') for filename in
                                        self.filenames_label_nib]
            self.num_samples = num_samples
        else:
            num_train = int(np.floor(self.train_percentage * num_samples))

            self.filenames_image_nib = [current_sample['image'] for current_sample in data_meta['training']][
                                       num_train:]
            self.filenames_label_nib = [current_sample['label'] for current_sample in data_meta['training']][
                                       num_train:]

            self.filenames_image_npy = [os.path.join('.', 'imagesTrNP', f'{filename[11:-7]}.npy') for filename in
                                        self.filenames_image_nib]
            self.filenames_label_npy = [os.path.join('.', 'labelsTrNP', f'{filename[11:-7]}.npy') for filename in
                                        self.filenames_label_nib]
            # self.num_samples = int(np.ceil((1 - self.train_percentage) * num_samples))
            self.num_samples = len(self.filenames_image_nib)
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


# MODEL DEEPMEDIC
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

class ResBlock(nn.Module):
    '''
        Adapted from: https://github.com/pykao/BraTS2018-tumor-segmentation/blob/master/models/deepmedic.py
    '''

    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()

        self.inplanes = inplanes
        self.conv1 = nn.Conv3d(inplanes, planes, 3, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        x = x[:, :, 2:-2, 2:-2, 2:-2]
        y[:, :self.inplanes] += x
        y = self.relu(y)
        return y


def conv3x3(inplanes, planes, ksize=3):
    return nn.Sequential(
        nn.Conv3d(inplanes, planes, ksize, bias=False),
        nn.BatchNorm3d(planes),
        nn.ReLU(inplace=True))


def repeat(x, n=3):
    # nc333
    b, c, h, w, t = x.shape
    x = x.unsqueeze(5).unsqueeze(4).unsqueeze(3)
    x = x.repeat(1, 1, 1, n, 1, n, 1, n)
    return x.view(b, c, n * h, n * w, n * t)


class DeepMedic(nn.Module):
    '''
        Adapted from: https://github.com/pykao/BraTS2018-tumor-segmentation/blob/master/models/deepmedic.py
    '''

    def __init__(self, input_channels=1, n1=30, n2=40, n3=50, m=150, up=True):
        super(DeepMedic, self).__init__()
        # n1, n2, n3 = 30, 40, 50
        num_classes = 3
        n = 2 * n3
        self.branch1 = nn.Sequential(
            conv3x3(input_channels, n1),
            conv3x3(n1, n1),
            ResBlock(n1, n2),
            ResBlock(n2, n2),
            ResBlock(n2, n3))

        self.branch2 = nn.Sequential(
            conv3x3(input_channels, n1),
            conv3x3(n1, n1),
            conv3x3(n1, n2),
            conv3x3(n2, n2),
            conv3x3(n2, n2),
            conv3x3(n2, n2),
            conv3x3(n2, n3),
            conv3x3(n3, n3))

        self.up3 = nn.Upsample(scale_factor=3, mode='trilinear', align_corners=False) if up else repeat

        self.fc = nn.Sequential(
            conv3x3(n, m, 1),
            conv3x3(m, m, 1),
            nn.Conv3d(m, num_classes, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x1, x2 = inputs
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)
        x2 = self.up3(x2)
        x = torch.cat([x1, x2], 1)
        x = self.fc(x)
        return x


# TRAIN/VAL/TEST
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

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


class ModelConainer():
    def __init__(self, params_model):
        self.params_model = params_model

    def __init_train_params(self):
        self.run_mode = 'train'

        self.loss_dict_train = {
            'total': [],
            'dice': [],
            'mse': [],
            'ce': [],
            'dice_n_mse': [],
            'dice_n_mse_n_ce': []
        }

        self.loss_dict_val = {
            'total': [],
            'dice': [],
            'mse': [],
            'ce': [],
            'dice_n_mse': [],
            'dice_n_mse_n_ce': []
        }

        self.loss_best_train = {
            'total': np.inf,
            'dice': np.inf,
            'mse': np.inf,
            'ce': np.inf,
            'dice_n_mse': np.inf,
            'dice_n_mse_n_ce': np.inf
        }

        self.loss_best_val = {
            'total': np.inf,
            'dice': np.inf,
            'mse': np.inf,
            'ce': np.inf,
            'dice_n_mse': np.inf,
            'dice_n_mse_n_ce': np.inf
        }

    def __init_inference_params(self):
        self.run_mode = 'inference'

    def __setup_logger(self):

        if not self.params_train['resume_condition']:
            reload(logging)

        logging.basicConfig(filename=self.params_train['path_logger_full'], encoding='utf-8', level=logging.DEBUG)

    def __create_params_file(self):
        params_dict = {
            'params_model': self.params_model,
            'params_train': self.params_train
        }

        params_dict = json.dumps(params_dict, indent=4, sort_keys=False)

        with open(self.params_train['path_params_full'], 'w') as outfile:
            outfile.write(params_dict)

        self.__print(f'{"*" * 100}')
        self.__print('\t\tTraining starting with params:')
        self.__print(f'{"*" * 100}')
        self.__print(f'{params_dict}')
        self.__print(f'{"*" * 100}')

    def __create_checkpoint_dir(self):
        if self.params_train['resume_condition']:
            self.params_train['dirname_checkpoint'] = self.params_train['resume_dir'][:11]
            self.params_train['path_checkpoint_full'] = self.params_train['resume_dir']
        else:
            self.params_train['dirname_checkpoint'] = f'{self.params_model["experiment_name"]}__' \
                                                      f'{self.params_model["init_timestamp"]}__' \
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

    def train(self, params_train, transform_train):
        self.params_train = params_train
        self.transform_train = transform_train
        self.__init_train_params()
        self.__create_checkpoint_dir()
        self.__create_params_file()
        self.__setup_logger()
        self.__fit_model()

    def inference(self, params_inference):

        self.params_inference = params_inference

        self.__init_inference_params()
        self.__run_inference()

    def __set_device(self):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def __get_dataloaders(self, run_mode):
        if run_mode == 'train':
            self.dataset_train = DatasetHepatic(
                run_mode='train',
                transform=self.transform_train,
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
                transform=None,
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
                shuffle=True,
                num_workers=self.params_train['num_workers'],
                pin_memory=self.params_train['pin_memory'],
                prefetch_factor=self.params_train['prefetch_factor'],
                persistent_workers=self.params_train['persistent_workers']
            )

            self.dataloader_val = DataLoader(
                self.dataset_val,
                batch_size=self.params_train['batch_size'],
                shuffle=False,
                num_workers=self.params_train['num_workers'],
                pin_memory=self.params_train['pin_memory'],
                prefetch_factor=self.params_train['prefetch_factor'],
                persistent_workers=self.params_train['persistent_workers']
            )

        elif run_mode == 'inference':
            self.dataset_inference = DatasetHepatic(
                run_mode='inference',
                transform=None,
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
                shuffle=False,
                num_workers=self.params_inference['num_workers'],
                pin_memory=self.params_inference['pin_memory'],
                prefetch_factor=self.params_inference['prefetch_factor'],
                persistent_workers=self.params_inference['persistent_workers']
            )

    def __define_model(self):
        if self.params_model['model_name'] == 'deep_medic':
            self.model = DeepMedic().to(self.device)

    def __define_criterions(self):
        self.criterion_mse = nn.MSELoss()
        self.criterion_dice = GeneralizedDiceLoss()
        self.criterion_ce = nn.CrossEntropyLoss()

    def __define_optimizr(self):
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

    def __define_lr_scheduler(self):
        if self.params_train['lr_scheduler_name'] == 'plateau':
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=self.params_train['patience_lr_scheduler'],
                factor=self.params_train['factor_lr_scheduler'],
                verbose=True)

    def __put_to_device(self, device, tensors):
        for index, tensor in enumerate(tensors):
            tensors[index] = tensor.to(device)
        return tensors

    def __get_one_hot_labels(self, input, labels, squeeze_dim=None):
        output = torch.zeros((input.shape[0], len(labels), input.shape[2], input.shape[3], input.shape[4]),
                             dtype=input.dtype).to(input.device)

        if not squeeze_dim is None:
            for index, label in enumerate(labels):
                output[:, index] = torch.where(input == label, 1, 0).squeeze(squeeze_dim)
        else:
            for index, label in enumerate(labels):
                output[:, label] = torch.where(input == label, 1, 0)

        return output

    def __criterion_generalized_dice(self, im_real, im_pred):
        '''
            Following the equation from https://arxiv.org/abs/1707.03237 page 3
        '''
        weights = torch.autograd.Variable(3, dtype=torch.float64, requires_grad=True)
        for index in range(3):
            count = torch.tensor(torch.sum(torch.where(im_real == index, 1, 0)), dtype=torch.double, requires_grad=True)
            # if none of the voxels are of the current category, set weight to 1
            if count == 0:
                weights[index] = torch.tensor(1, dtype=torch.double, requires_grad=True)
            else:
                weights[index] = 1 / count ** 2

        numerator = torch.zeros(3, dtype=torch.double, requires_grad=True)
        denominator = torch.zeros(3, dtype=torch.double, requires_grad=True)

        for index in range(3):
            r_l_n = torch.where(im_real == index, 1, 0)
            p_l_n = torch.where(im_pred == index, 1, 0)

            # numerator
            mult = r_l_n * p_l_n
            numerator[index] = weights[index] * torch.sum(mult)

            current_denominator = weights[index] * (torch.sum(r_l_n) + torch.sum(p_l_n))
            denominator[index] = current_denominator

        dice_loss = 1 - (2 * torch.sum(numerator) / torch.sum(denominator))

        return dice_loss

    def __save_model(self):
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

            if self.loss_dict_val['total'][-1] <= min(self.loss_dict_val['total']):
                save_path = os.path.join(self.params_train['path_checkpoint_full'], f'best.pth')
                torch.save(save_dict, save_path)
                self.__print(f'{"*" * 10}\tNew best model saved at:\t{self.index_epoch + 1}\t{"*" * 10}')

    def __load_model(self):
        '''
            Loads the model
        '''
        if self.run_mode == 'train':
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

                self.__print(f'Model loaded from epoch:\t{self.index_epoch}')
                self.index_epoch += 1
                self.start_epoch = self.index_epoch
                self.__print(f'Resuming training from epoch:\t{self.index_epoch}')

        elif self.run_mode == 'inference':
            filename_checkpoint = f'{self.params_inference["resume_epoch"]}.pth'
            load_path = os.path.join(self.params_inference['path_checkpoint'],
                                     self.params_inference['resume_dir'],
                                     filename_checkpoint)

            if not os.path.exists(load_path):
                raise FileNotFoundError(f'File {load_path} doesn\'t exist')

            checkpoint = torch.load(load_path)
            self.index_epoch = checkpoint['index_epoch']
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.params_model = checkpoint['params_model']
            self.index_epoch += 1
            self.__print(f'Model loaded from epoch:\t{self.index_epoch}')

    def __early_stop(self):
        '''
            If early stopping condition meets, break training
        '''
        # train for at least self.params_train['min_epochs_to_train'] epochs
        if self.index_epoch > self.params_train['min_epochs_to_train']:

            # if the latest loss is lower than the best, continue training,
            # otherwise check the last x losses loss_dict_val
            # if self.loss_best_val['total'][-1] < min(self.loss_best_val['total']):
            if self.loss_dict_val['total'][-1] < min(self.loss_dict_val['total']):
                self.break_training_condition = False
            else:
                index_start = len(self.loss_dict_val['total']) - 1
                index_stop = len(self.loss_dict_val['total']) - 1 - self.params_train['patience_early_stop']

                # if any of the last x losses are greater than the best loss, increase counter
                counter = 0
                for index in range(index_start, index_stop, -1):
                    if self.loss_dict_val['total'][index] > min(self.loss_dict_val['total']):
                        counter += 1

                # if counter equals the patience, break training
                if counter >= self.params_train['patience_early_stop']:
                    self.__print(f'Early stopping at epoch:\t{self.index_epoch + 1}')
                    self.break_training_condition = True

    def __run_epoch(self, dataloader, run_mode):
        '''
            Runs one epoch of training and validation loops
        '''
        loss_list_total = []
        loss_list_dice = []
        loss_list_ce = []
        loss_list_mse = []
        loss_list_dice_n_mse = []
        loss_list_dice_n_mse_n_ce = []

        loss_total, loss_dice, loss_mse, loss_ce, loss_dice_n_mse, loss_dice_n_mse_n_ce = np.Inf, np.Inf, np.Inf, np.Inf, np.Inf, np.inf

        for index_batch, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
            if run_mode == 'train':
                self.model.train()
                self.optimizer.zero_grad()
            else:
                self.model.eval()

            (image_patch_normal, image_patch_low_up, label_patch_out_real) = self.__put_to_device(self.device, batch)

            if self.params_train['batch_size_inner'] > 1:
                if len(image_patch_normal.shape) == 6:
                    batch_size_stacked = image_patch_normal.shape[0] * image_patch_normal.shape[1]

                    image_patch_normal = image_patch_normal.reshape(batch_size_stacked, image_patch_normal.shape[2],
                                                                    image_patch_normal.shape[3],
                                                                    image_patch_normal.shape[4],
                                                                    image_patch_normal.shape[5])
                    image_patch_low_up = image_patch_low_up.reshape(batch_size_stacked, image_patch_low_up.shape[2],
                                                                    image_patch_low_up.shape[3],
                                                                    image_patch_low_up.shape[4],
                                                                    image_patch_low_up.shape[5])
                    label_patch_out_real = label_patch_out_real.reshape(batch_size_stacked,
                                                                        label_patch_out_real.shape[2],
                                                                        label_patch_out_real.shape[3],
                                                                        label_patch_out_real.shape[4],
                                                                        label_patch_out_real.shape[5])

                    image_patch_low = torch.zeros((image_patch_low_up.shape[0],
                                                   self.params_model['patch_size_low'],
                                                   self.params_model['patch_size_low'],
                                                   self.params_model['patch_size_low'])).to(self.device)

                else:
                    image_patch_normal, image_patch_low_up, label_patch_out_real = image_patch_normal.squeeze(
                        0), image_patch_low_up.squeeze(0), label_patch_out_real.squeeze(0)

                image_patch_low = torch.zeros((image_patch_low_up.shape[0],
                                               image_patch_low_up.shape[1],
                                               self.params_model['patch_size_low'],
                                               self.params_model['patch_size_low'],
                                               self.params_model['patch_size_low'])).to(self.device)

                for index, current_low_up in enumerate(image_patch_low_up):
                    current_low = F.avg_pool3d(input=current_low_up, kernel_size=3, stride=None)
                    image_patch_low[index] = copy.deepcopy(current_low.detach())

            # forward pass
            label_patch_out_pred = self.model.forward((image_patch_normal, image_patch_low))

            # convert label_patch_out_real to one hot
            label_patch_out_real_one_hot = self.__get_one_hot_labels(label_patch_out_real, labels=[0, 1, 2],
                                                                     squeeze_dim=1)

            # generalized dice loss_total # dice, mse, dice_n_mse
            if self.params_train['loss_name'] == 'dice':
                loss_dice = self.criterion_dice(F.softmax(label_patch_out_pred.float(), dim=1),
                                                label_patch_out_real_one_hot.float())
                loss_list_dice.append(loss_dice.item())
                # print(f'loss_dice:\t{loss_dice.item():.5f}')
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
            elif self.params_train['loss_name'] == 'dice_n_mse_n_ce':
                loss_dice = self.criterion_dice(F.softmax(label_patch_out_pred.float(), dim=1),
                                                label_patch_out_real_one_hot.float())
                loss_mse = self.criterion_mse(F.softmax(label_patch_out_pred.float(), dim=1),
                                              label_patch_out_real_one_hot.float())
                loss_ce = self.criterion_ce(label_patch_out_pred.float(), label_patch_out_real.squeeze(1).long())

                loss_dice_n_mse_n_ce = loss_dice + loss_mse + loss_ce

                loss_total = loss_dice_n_mse_n_ce
                loss_list_dice.append(loss_dice.item())
                loss_list_mse.append(loss_mse.item())
                loss_list_ce.append(loss_ce.item())
                loss_list_dice_n_mse_n_ce.append(loss_dice_n_mse_n_ce.item())
            else:
                raise NotImplementedError(f'Invalid criterion selected:\t{self.params_train["loss_name"]}')

            loss_list_total.append(loss_total)

            if run_mode == 'train':
                # calculate gradients and update weights
                loss_total.backward()
                self.optimizer.step()
            sep = '\t' if run_mode == 'train' else '\t\t'
            # self.__print(f'\tBatch:\t[{index_batch + 1} / {len(dataloader)}]'
            #                f'\n\t\t{str(run_mode).upper()}{sep}-->\t\tLoss ({self.params_train["loss_name"]}):\t\t{loss_total.item():.5f}')

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
            # break

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
        if len(loss_list_dice_n_mse_n_ce) > 0:
            loss_dice_n_mse_n_ce = sum(loss_list_dice_n_mse_n_ce) / len(loss_list_dice_n_mse_n_ce)

        if run_mode == 'train':
            self.loss_dict_train['total'].append(loss_total.item())
            self.loss_dict_train['dice'].append(loss_dice)
            self.loss_dict_train['mse'].append(loss_mse)
            self.loss_dict_train['ce'].append(loss_ce)
            self.loss_dict_train['dice_n_mse'].append(loss_dice_n_mse)
            self.loss_dict_train['dice_n_mse_n_ce'].append(loss_dice_n_mse_n_ce)

        elif run_mode == 'val':
            self.loss_dict_val['total'].append(loss_total.item())
            self.loss_dict_val['dice'].append(loss_dice)
            self.loss_dict_val['mse'].append(loss_mse)
            self.loss_dict_val['ce'].append(loss_ce)
            self.loss_dict_val['dice_n_mse'].append(loss_dice_n_mse)
            self.loss_dict_val['dice_n_mse_n_ce'].append(loss_dice_n_mse_n_ce)

    def __update_best_losses(self):
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

    def __print(self, message):
        print(message)
        logging.debug(message)
        logging.debug('working')

    def __fit_model(self):
        '''
            Trains and validates a model given hyperparameters, model and optimizer name, dataloaders, num_epochs
        '''

        # set up dataloaders, model, criterions, optimizers, schedulers
        self.__set_device()
        self.__get_dataloaders('train')
        self.__define_model()
        self.__define_criterions()
        self.__define_optimizr()
        self.__define_lr_scheduler()

        # variables to keep track of training progress
        self.start_epoch = 0
        self.break_training_condition = False
        self.end_epoch = self.params_train['num_epochs']

        self.__load_model()

        for index_epoch in range(self.start_epoch, self.end_epoch):
            time_start = time.time()
            self.index_epoch = index_epoch

            # train
            self.__run_epoch(
                dataloader=self.dataloader_train,
                run_mode='train'
            )

            # validation
            with torch.no_grad():
                self.__run_epoch(
                    dataloader=self.dataloader_val,
                    run_mode='val'
                )

            # choose which loss to put into the scheduler
            self.lr_scheduler.step(self.loss_dict_val['total'][-1])

            duration = time.time() - time_start

            self.__print(f'\n{"-" * 100}'
                         f'\nEpoch:\t[{index_epoch + 1} / {self.end_epoch}]\t\t'
                         f'Time:\t{duration:.2f} s'
                         f'\n\tTRAIN\t\t-->\t\tLoss Total:\t\t{self.loss_dict_train["total"][-1]:.5f}'
                         f'\n\tVAL\t\t\t-->\t\tLoss Total:\t\t{self.loss_dict_val["total"][-1]:.5f}'
                         f'\t\tBest:\t{min(self.loss_dict_val["total"]):.5f}'
                         f'\n{"-" * 100}\n')

            self.__update_best_losses()
            self.__save_model()
            self.__early_stop()
            # self.__early_stopper()

            if self.break_training_condition:
                break

    def __run_inference(self):
        '''
            Run 3D inference on the validation set. Generates a 3D volume of predicted labels with same shape as the original one
        '''
        self.__set_device()
        self.__get_dataloaders('inference')
        self.__define_model()
        self.__define_criterions()
        self.__load_model()

        self.__print(f'{"*" * 100}')
        self.__print('\t\tInference starting with params:')
        self.__print(f'{"*" * 100}')
        params_dict = {
            'params_model': self.params_model,
            'params_inference': self.params_inference
        }
        params_dict = json.dumps(params_dict, indent=4, sort_keys=False)
        self.__print(f'{params_dict}')
        self.__print(f'{"*" * 100}')

        for index_batch, batch in enumerate(self.dataloader_inference):
            images, labels_real, index_filename = batch
            (images, labels_real) = self.__put_to_device(self.device, [images, labels_real])

            labels_pred, labels_pred_probabilistic, loss_dice, loss_mse = self.__stride_depth_and_inference(
                images_real=images,
                labels_real=labels_real
            )
            # assuming the batch size == 1
            index_filename = index_filename[0]
            self.__print(
                f'{index_batch + 1}: \tprediction_{index_filename}.npy\tLoss DICE:\t{loss_dice:.5f}\tLoss MSE:\t{loss_mse:.5f}')

            labels_real, labels_pred, labels_pred_probabilistic = labels_real.cpu().detach().numpy(), labels_pred.cpu().detach().numpy(), labels_pred_probabilistic.cpu().detach().numpy()

            predictions_path = os.path.join('predictions', 'abhijit')
            os.makedirs(predictions_path, exist_ok=True)
            # saves the probabilistic outputs (bs x 3 x h x w x d)
            np.save(os.path.join(predictions_path, f'prediction_{index_filename}'), labels_pred_probabilistic,
                    allow_pickle=True)

    def __stride_depth_and_inference(self, images_real, labels_real):
        self.model.eval()
        patch_size_normal = self.params_model['patch_size_normal']
        patch_size_low = self.params_model['patch_size_low']
        patch_size_out = self.params_model['patch_size_out']
        patch_low_factor = self.params_model['patch_low_factor']

        with torch.no_grad():
            loss_list_dice = []
            loss_list_mse = []

            device = images_real.device
            batch_size, height, width, depth = images_real.shape

            # --------- loop through the whole image volume
            patch_size_low_up = patch_size_low * patch_low_factor

            patch_half_normal = (patch_size_normal - 1) // 2
            patch_half_low = (patch_size_low - 1) // 2
            patch_half_low_up = (patch_size_low_up - 1) // 2
            patch_half_out = (patch_size_out - 1) // 2

            height_new = height + patch_size_low_up
            width_new = width + patch_size_low_up
            depth_new = depth + patch_size_low_up

            # create a placeholder for the padded image
            images_padded = torch.zeros((batch_size, height_new, width_new, depth_new), dtype=torch.float32).to(device)
            labels_real_padded = torch.zeros((batch_size, height_new, width_new, depth_new), dtype=torch.float32).to(
                device)

            # labels_padded = torch.zeros((batch_size, height_new, width_new, depth_new), dtype=torch.float32).to(device)
            # print(f'images_real.shape:\t{images_real.shape}')
            # print(f'images_padded.shape:\t{images_padded.shape}')

            # copy the original image to the placeholder
            images_padded[
            :,
            patch_half_low_up: height + patch_half_low_up,
            patch_half_low_up: width + patch_half_low_up,
            patch_half_low_up: depth + patch_half_low_up
            ] = copy.deepcopy(images_real).to(device)

            labels_real_padded[
            :,
            patch_half_low_up: height + patch_half_low_up,
            patch_half_low_up: width + patch_half_low_up,
            patch_half_low_up: depth + patch_half_low_up
            ] = copy.deepcopy(labels_real).to(device)

            # print(f'{patch_half_low_up} -> {height + patch_half_low_up}')
            # print(f'{patch_half_low_up} -> {width + patch_half_low_up}')
            # print(f'{patch_half_low_up} -> {depth + patch_half_low_up}')

            # placeholder to store the inferred/reconstructed image labels
            labels_pred_whole_image = torch.zeros_like(images_real).to(device)
            labels_pred_whole_image_probabilistic = torch.zeros((batch_size, 3, height, width, depth),
                                                                dtype=torch.float32).to(device)
            # print(f'labels_pred_whole_image.shape:\t{labels_pred_whole_image.shape}')

            # indices of the original image
            h_start_orig = 0
            h_end_orig = h_start_orig + patch_size_out

            for index_h in tqdm(range(patch_half_low_up, height_new - patch_half_low_up, patch_size_out),
                                leave=False):
                # print(index_h)
                h_start_normal = index_h - patch_half_normal
                h_end_normal = index_h + patch_half_normal + 1

                h_start_low_up = index_h - patch_half_low_up
                h_end_low_up = index_h + patch_half_low_up + 1

                h_start_out = index_h - patch_half_out
                h_end_out = index_h + patch_half_out + 1

                # if the starting index of the out height > padded height; break
                if h_end_out > height_new:
                    break

                w_start_orig = 0
                w_end_orig = w_start_orig + patch_size_out

                for index_w in range(patch_half_low_up, width_new - patch_half_low_up, patch_size_out):

                    w_start_normal = index_w - patch_half_normal
                    w_end_normal = index_w + patch_half_normal + 1

                    w_start_low_up = index_w - patch_half_low_up
                    w_end_low_up = index_w + patch_half_low_up + 1

                    w_start_out = index_w - patch_half_out
                    w_end_out = index_w + patch_half_out + 1

                    if w_end_out > width_new:
                        break

                    d_start_orig = 0
                    d_end_orig = d_start_orig + patch_size_out

                    for index_d in range(patch_half_low_up, depth_new - patch_half_low_up, patch_size_out):

                        d_start_normal = index_d - patch_half_normal
                        d_end_normal = index_d + patch_half_normal + 1

                        d_start_low_up = index_d - patch_half_low_up
                        d_end_low_up = index_d + patch_half_low_up + 1

                        d_start_out = index_d - patch_half_out
                        d_end_out = index_d + patch_half_out + 1

                        if d_end_out > depth_new:
                            break

                        # extract the current patch of the expanded image
                        image_patch_normal = images_padded[
                                             :,
                                             h_start_normal: h_end_normal,
                                             w_start_normal: w_end_normal,
                                             d_start_normal: d_end_normal
                                             ]
                        # print('\nNormal')
                        # print(f'{h_start_normal} -> {h_end_normal}')
                        # print(f'{w_start_normal} -> {w_end_normal}')
                        # print(f'{d_start_normal} -> {d_end_normal}')

                        image_patch_low_up = images_padded[
                                             :,
                                             h_start_low_up: h_end_low_up,
                                             w_start_low_up: w_end_low_up,
                                             d_start_low_up: d_end_low_up
                                             ]

                        # print('\nlow_up')
                        # print(f'{h_start_low_up} -> {h_end_low_up}')
                        # print(f'{w_start_low_up} -> {w_end_low_up}')
                        # print(f'{d_start_low_up} -> {d_end_low_up}')

                        # extract the current output patch of the expanded label
                        label_patch_out_real = labels_real_padded[
                                               :,
                                               h_start_out: h_end_out,
                                               w_start_out: w_end_out,
                                               d_start_out: d_end_out
                                               ]
                        # print('\nout')
                        # print(f'{h_start_out} -> {h_end_out}')
                        # print(f'{w_start_out} -> {w_end_out}')
                        # print(f'{d_start_out} -> {d_end_out}')

                        # if d_start_out == 42:
                        #     print('sssss')

                        if not (label_patch_out_real.shape[1] * label_patch_out_real.shape[2] *
                                label_patch_out_real.shape[3] > 0):
                            # print('here')
                            continue

                        # pad uneven images (image patch normal)
                        image_patch_normal_temp = torch.zeros(
                            (batch_size, patch_size_normal, patch_size_normal, patch_size_normal)).to(device)
                        image_patch_normal_temp[:, :image_patch_normal.shape[1], :image_patch_normal.shape[2],
                        :image_patch_normal.shape[3]] = image_patch_normal
                        image_patch_normal = image_patch_normal_temp

                        # pad uneven images (image patch low_up)
                        image_patch_low_up_temp = torch.zeros(
                            (batch_size, patch_size_low_up, patch_size_low_up, patch_size_low_up)).to(device)
                        image_patch_low_up_temp[:, :image_patch_low_up.shape[1], :image_patch_low_up.shape[2],
                        :image_patch_low_up.shape[3]] = image_patch_low_up
                        image_patch_low_up = image_patch_low_up_temp

                        # resize (downsample) image_patch_low
                        image_patch_low = F.avg_pool3d(input=image_patch_low_up, kernel_size=3, stride=None)

                        # perform forward pass
                        label_patch_out_pred = self.model.forward(
                            (image_patch_normal.unsqueeze(0), image_patch_low.unsqueeze(0)))

                        # print(label_patch_out_real.shape)
                        # clip extra parts
                        if label_patch_out_real.shape[1] < patch_size_out:
                            label_patch_out_pred = label_patch_out_pred[:, :, :label_patch_out_real.shape[1], :, :]

                        if label_patch_out_real.shape[2] < patch_size_out:
                            label_patch_out_pred = label_patch_out_pred[:, :, :, :label_patch_out_real.shape[2], :]

                        if label_patch_out_real.shape[3] < patch_size_out:
                            label_patch_out_pred = label_patch_out_pred[:, :, :, :, :label_patch_out_real.shape[3]]

                        # # remove any dimensions with 0 elements
                        # if (label_patch_out_pred.shape[2] == 0) or (label_patch_out_pred.shape[3] == 0) or (label_patch_out_pred.shape[4] == 0) or (
                        #         label_patch_out_real.shape[2] == 0) or (label_patch_out_real.shape[3] == 0) or (
                        #         label_patch_out_real.shape[4] == 0):
                        #     break

                        # print(label_patch_out_pred.shape)
                        # convert label_patch_out_real to one hot
                        label_patch_out_real_one_hot = torch.zeros_like(label_patch_out_pred).to(device)
                        # print(label_patch_out_real_one_hot.shape)
                        label_patch_out_real_one_hot[:, 0] = torch.where(label_patch_out_real == 0, 1, 0)
                        label_patch_out_real_one_hot[:, 1] = torch.where(label_patch_out_real == 1, 1, 0)
                        label_patch_out_real_one_hot[:, 2] = torch.where(label_patch_out_real == 2, 1, 0)

                        # cross-entropy loss_dice
                        loss_dice = self.criterion_dice(F.softmax(label_patch_out_pred.float(), dim=1),
                                                        label_patch_out_real_one_hot.float())
                        loss_mse = self.criterion_mse(F.softmax(label_patch_out_pred.float(), dim=1),
                                                      label_patch_out_real_one_hot.float())
                        # print(loss_mse.item())
                        loss_list_dice.append(loss_dice)
                        loss_list_mse.append(loss_mse)
                        # print(loss_dice)

                        label_patch_out_pred_double = torch.argmax(label_patch_out_pred.detach(), dim=1)
                        label_patch_out_pred_double_temp = torch.zeros(batch_size, patch_size_out, patch_size_out,
                                                                       patch_size_out).to(device)
                        label_patch_out_pred_double_temp[:, :label_patch_out_pred_double.shape[1],
                        :label_patch_out_pred_double.shape[2],
                        :label_patch_out_pred_double.shape[3]] = label_patch_out_pred_double
                        label_patch_out_pred_double = label_patch_out_pred_double_temp

                        bs, h, w, d = labels_pred_whole_image[:, h_start_orig: h_end_orig,
                                      w_start_orig: w_end_orig,
                                      d_start_orig: d_end_orig].shape

                        # save the pixel wise predictions
                        labels_pred_whole_image[:, h_start_orig: h_end_orig, w_start_orig: w_end_orig,
                        d_start_orig: d_end_orig] = label_patch_out_pred_double[:, :h, :w, :d].detach()

                        # save the probabilistic predictions
                        labels_pred_whole_image_probabilistic[:, :, h_start_orig: h_end_orig, w_start_orig: w_end_orig,
                        d_start_orig: d_end_orig] = label_patch_out_pred[:, :, :h, :w, :d].detach()

                        d_start_orig = d_start_orig + patch_size_out
                        d_end_orig = d_end_orig + patch_size_out

                    w_start_orig = w_start_orig + patch_size_out
                    w_end_orig = w_end_orig + patch_size_out

                h_start_orig = h_start_orig + patch_size_out
                h_end_orig = h_end_orig + patch_size_out

                loss_dice = sum(loss_list_dice) / (len(loss_list_dice) + 1e-9)
                loss_mse = sum(loss_list_mse) / (len(loss_list_mse) + 1e-9)

        return labels_pred_whole_image, labels_pred_whole_image_probabilistic, loss_dice, loss_mse


if __name__ == '__main__':
    params_model = {
        'experiment_name': 'step_1',
        'model_name': 'deep_medic',
        'patch_size_normal': 25,
        'patch_size_low': 19,
        'patch_size_out': 9,
        'patch_low_factor': 3,
        'run_mode': None,
        'dataset_variant': 'npy',  # npy, nib
        'create_numpy_dataset': False,
        'init_timestamp': datetime.now().strftime("%H-%M-%S__%d-%m-%Y")
    }

    params_train = {
        'optimizer_name': 'adam',  # adam, sgd_w_momentum
        'loss_name': 'dice',  # dice, mse, ce, dice_n_mse, dice_n_mse_n_ce
        'beta_1': 0.9,
        'beta_2': 0.999,
        'momentum': 0.9,
        'use_amsgrad': True,
        'learning_rate': 0.0002,  # 0.0002
        'lr_scheduler_name': 'plateau',
        'patience_lr_scheduler': 2,
        'factor_lr_scheduler': 0.1,
        'early_stop_condition': True,
        'patience_early_stop': 5,
        'early_stop_patience_counter': 0,
        'min_epochs_to_train': 10,
        'num_epochs': 100,
        'save_every_epoch': True,

        'save_condition': True,  # whether to save the model
        'resume_condition': False,  # whether to resume training

        'resume_dir': '14-10-08__01-04-2022__deep_medic__dice__adam__lr_0.0001__ep_50',
        'resume_epoch': 'latest',

        'batch_size': 8,  # 8
        'batch_size_inner': 16,  # 16 (how many patches to generate per sample)
        'train_percentage': 0.8,
        'num_workers': 8,  # 8
        'pin_memory': True,
        'prefetch_factor': 16,
        'persistent_workers': True,

        'path_checkpoint': os.path.join('.', 'checkpoints'),
        'path_checkpoint_full': '',
        'dirname_checkpoint': '',
        'filename_params': 'params.json',
        'filename_logger': 'logger.txt',
        'path_params_full': '',
        'path_logger_full': ''
    }

    # instanciate model
    set_seed(1)
    model_container = ModelConainer(params_model)

    # basic transformations for part 1
    transform_train = transforms.Compose([
        ToTorchTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # train the model
    model_container.train(params_train=params_train, transform_train=transform_train)

    # params_inference = {
    #     'loss_name': 'dice',
    #     'batch_size': 1,
    #     'train_percentage': 0.8,
    #     'num_workers': 1,  # 8
    #     'pin_memory': True,
    #     'prefetch_factor': 2,
    #     'persistent_workers': True,
    #
    #     'resume_dir': 'step_1__10-18-19__04-04-2022__deep_medic__dice__adam__lr_0.0002__ep_100',
    #     'resume_epoch': 'best',
    #     'path_checkpoint': os.path.join('.', 'checkpoints'),
    #     'path_checkpoint_full': '',
    #     'dirname_checkpoint': '',
    # }
    #
    # model_container.inference(params_inference=params_inference)
