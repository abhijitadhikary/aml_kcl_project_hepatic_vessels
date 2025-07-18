# import json
# import os
# import sys
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# # % matplotlib inline
# import nibabel as nib
# from tqdm import tqdm
# import json
#
# import torch
# import torchvision
# import torch.optim as optim
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
#
#
#
# class ToTorchTensor():
#     '''
#         Transforms a numpy ndarray to a torch tensor of the supplied datatype
#     '''
#     def __call__(self, input, datatype=torch.float32):
#         return torch.tensor(input, dtype=datatype)
#
# class DatasetHepatic(Dataset):
#     '''
#         min = 24
#         max = 181
#         median = 49
#         mean = 69
#     '''
#     def __init__(self, run_mode='train', transform=None, patch_size=16, label_percentage=0.1, use_probabilistic=False):
#         self.run_mode = run_mode
#         self.patch_size = patch_size
#         self.label_percentage = label_percentage
#         self.use_probabilistic = use_probabilistic
#         self.fetch_filenames()
#
#         self.transform = transform if not transform is None else transforms.Compose([ToTorchTensor()])
#
#     def __getitem__(self, index):
#         if self.run_mode in ['train', 'val']:
#             image = self.read_file_nib(self.filenames_image[index])
#             label = self.read_file_nib(self.filenames_label[index])
#
#             label_patch = self.get_random_patch(label)
#             image_patch = self.get_3D_crop(image, self.coordinate)
#
#             image_patch = self.transform(image_patch)
#             label_patch = torch.tensor(label_patch, dtype=torch.int64)
#
#         # if not self.run_mode == 'test':
#         #     label = self.read_file_nib(self.filenames_label[index])
#
#         return image_patch, label_patch
#
#
#     def __len__(self):
#         return self.num_samples
#
#     def get_label_percentage(self, input, label):
#         '''
#             Returns the percentage of supplied label in the voxel
#         '''
#         eps = 1e-9
#         denominator = input.shape[0] * input.shape[1] * input.shape[2]
#         numerator = np.sum(np.where(input == label, 1, 0))
#
#         return numerator / (denominator + eps)
#
#     def get_rand_index_3D(self, height=512, width=512, depth=20):
#         '''
#             Returns a random starting index (top-left) of a valid 3D volume
#         '''
#
#         index_h = np.random.randint(0, height - self.patch_size)
#         index_w = np.random.randint(0, width - self.patch_size)
#         index_d = np.random.randint(0, depth - self.patch_size)
#
#         return (index_h, index_w, index_d)
#
#     def get_3D_crop(self, input, coordinate):
#         '''
#             Returns a 3D patch of an input 3D image given a valid top-left coordinate
#         '''
#         return input[
#                coordinate[0]:coordinate[0]+self.patch_size,
#                coordinate[1]:coordinate[1]+self.patch_size,
#                coordinate[2]:coordinate[2]+self.patch_size,
#                ]
#
#     def set_probabilistic_label(self):
#         '''
#             Randomly with equal probability select one of the three labels to be the current label
#         '''
#         label_probability = np.random.rand()
#         if label_probability > 0.66:
#             self.current_selected_label = 2
#         elif label_probability < 0.33:
#             self.current_selected_label = 1
#         else:
#             self.current_selected_label = 0
#
#     def get_random_patch(self, input):
#         '''
#             Returns a valid cubic sub-volume with edge lenth = patch_size from a supplied 3D input volume image_input
#         '''
#         if len(input.shape) == 3:
#             height, width, depth = input.shape
#
#         loop_condition = True
#         if self.use_probabilistic:
#             self.set_probabilistic_label()
#         # keep sampling a new patch until the current label meets the desired overall percentage
#         while loop_condition:
#             # get a valid coordinate and extract the patch
#             self.coordinate = self.get_rand_index_3D(height, width, depth)
#             patch = self.get_3D_crop(input, self.coordinate)
#
#             if self.use_probabilistic:
#                 # get the percentage of the current label compared to the whole patch
#                 self.label_percentage_current = self.get_label_percentage(patch, self.current_selected_label)
#                 if self.label_percentage_current > self.label_percentage:
#                     loop_condition = False
#             else:
#                 loop_condition = False
#         return patch
#
#     def read_file_nib(self, filename):
#         '''
#             Reads a nibabel file and returns it in numpy ndarray format
#         '''
#         try:
#             data_nib = nib.load(filename).get_fdata()
#         except FileNotFoundError:
#             print(f'Error reading file: {filename}')
#
#         return data_nib
#
#     def fetch_filenames(self, path_meta='dataset.json'):
#         '''
#             Reads the dataset.json file and extracts the training and test image and/or labels
#         :return:
#         '''
#         try:
#             with open(path_meta) as file_meta:
#                 data_meta = json.loads(file_meta.read())
#         except FileNotFoundError:
#             print(f'Meta file: {self.path_meta} not found')
#
#         if self.run_mode == 'train':
#             self.filenames_image = [current_sample['image'] for current_sample in data_meta['training']]
#             self.filenames_label = [current_sample['label'] for current_sample in data_meta['training']]
#
#             if (not len(self.filenames_image) == len(self.filenames_label)):
#                 raise Exception('Inconsistent training image/label combination')
#             if len(self.filenames_image) == 0:
#                 raise Exception(f'Error reading {self.run_mode} images')
#             if len(self.filenames_label) == 0:
#                 raise Exception(f'Error reading {self.run_mode} labels')
#
#         elif self.run_mode == 'test':
#             self.filenames_image = [current_sample for current_sample in data_meta['test']]
#             if len(self.filenames_image) == 0:
#                 raise Exception(f'Error reading {self.run_mode} images')
#
#         self.num_samples = len(self.filenames_image)
#
#
#
#
