import copy
import torch
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

path_meta = 'dataset.json'

file_meta = open(path_meta)
data_meta = json.loads(file_meta.read())
file_meta.close()


# images = torch.tensor(a).unsqueeze(0)
a = nib.load(data_meta['training'][0]['label']).get_fdata()

# images = torch.tensor(a).unsqueeze(0)
images = torch.tensor(a).unsqueeze(0)[:, 200:256, 200:250]

def stride_depth_and_inference(model, optimizer, criterion_dice, images_real, labels_real, patch_size_normal=25, patch_size_low=19, patch_size_out=9, patch_low_factor=3):

    model.eval()

    with torch.no_grad():
        loss_list = []

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
        # labels_padded = torch.zeros((batch_size, height_new, width_new, depth_new), dtype=torch.float32).to(device)

        # copy the original image to the placeholder
        images_padded[
            :,
            patch_half_low_up: height + patch_half_low_up,
            patch_half_low_up: width + patch_half_low_up,
            patch_half_low_up: depth + patch_half_low_up
        ] = copy.deepcopy(images_real).to(device)

        # placeholder to store the inferred/reconstructed image labels
        labels_pred_whole_image = torch.zeros_like(images_real).to(device)

        # indices of the original image
        h_start_orig = 0
        h_end_orig = h_start_orig + patch_size_out

        for index_h in tqdm(range(patch_half_low_up + patch_half_out, height_new - patch_half_out, patch_size_out), leave=False):

            h_start_normal = index_h - patch_half_normal
            h_end_normal = index_h + patch_half_normal + 1

            h_start_low_up = index_h - patch_half_low_up
            h_end_low_up = index_h + patch_half_low_up + 1

            h_start_out = index_h - patch_half_out
            h_end_out = index_h + patch_half_out + 1

            if h_end_out > height_new:
                break

            w_start_orig = 0
            w_end_orig = w_start_orig + patch_size_out

            for index_w in range(patch_half_low_up + patch_half_out, width_new - patch_half_out, patch_size_out):

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

                for index_d in range(patch_half_low_up + patch_half_out, depth_new - patch_half_out,
                                     patch_size_out):

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

                    image_patch_low_up = images_padded[
                                :,
                                h_start_low_up: h_end_low_up,
                                w_start_low_up: w_end_low_up,
                                d_start_low_up: d_end_low_up
                    ]

                    # extract the current output patch of the expanded label
                    label_patch_out_real = labels_real[
                                :,
                                h_start_out: h_end_out,
                                w_start_out: w_end_out,
                                d_start_out: d_end_out
                    ]

                    # pad uneven images
                    image_patch_normal_temp = torch.zeros((batch_size, patch_size_normal, patch_size_normal, patch_size_normal)).to(device)
                    image_patch_normal_temp[:, :image_patch_normal.shape[1], :image_patch_normal.shape[2], :image_patch_normal.shape[3]] = image_patch_normal
                    image_patch_normal = image_patch_normal_temp

                    image_patch_low_up_temp = torch.zeros((batch_size, patch_size_low_up, patch_size_low_up, patch_size_low_up)).to(device)
                    image_patch_low_up_temp[:, :image_patch_low_up.shape[1], :image_patch_low_up.shape[2], :image_patch_low_up.shape[3]] = image_patch_low_up
                    image_patch_low_up = image_patch_low_up_temp

                    # resize (downsample) image_patch_low
                    image_patch_low = F.avg_pool3d(input=image_patch_low_up, kernel_size=3, stride=None)

                    # perform forward pass
                    label_patch_out_pred = model.forward((image_patch_normal.unsqueeze(0), image_patch_low.unsqueeze(0)))

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

                    # cross-entropy loss
                    loss = criterion_dice(label_patch_out_pred.float(), label_patch_out_real_one_hot)
                    loss_list.append(loss)
                    # print(loss)

                    label_patch_out_pred_double = torch.argmax(label_patch_out_pred.detach(), dim=1)
                    label_patch_out_pred_double_temp = torch.zeros(batch_size, patch_size_out, patch_size_out, patch_size_out).to(device)
                    label_patch_out_pred_double_temp[:, :label_patch_out_pred_double.shape[1], :label_patch_out_pred_double.shape[2], :label_patch_out_pred_double.shape[3]] = label_patch_out_pred_double
                    label_patch_out_pred_double = label_patch_out_pred_double_temp

                    bs, h, w, d = labels_pred_whole_image[:, h_start_orig: h_end_orig, w_start_orig: w_end_orig, d_start_orig: d_end_orig].shape
                    labels_pred_whole_image[:, h_start_orig: h_end_orig, w_start_orig: w_end_orig, d_start_orig: d_end_orig] = label_patch_out_pred_double[:, :h, :w, :d]

                    d_start_orig = d_start_orig + patch_size_out
                    d_end_orig = d_end_orig + patch_size_out

                w_start_orig = w_start_orig + patch_size_out
                w_end_orig = w_end_orig + patch_size_out

            h_start_orig = h_start_orig + patch_size_out
            h_end_orig = h_end_orig + patch_size_out

            loss = sum(loss_list) / len(loss_list)
    return labels_pred_whole_image, model, optimizer, criterion_dice, loss