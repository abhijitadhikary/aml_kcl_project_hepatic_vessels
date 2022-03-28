import copy
import torch
import json
import numpy as np
import nibabel as nib


path_meta = 'dataset.json'

file_meta = open(path_meta)
data_meta = json.loads(file_meta.read())
file_meta.close()


# images = torch.tensor(a).unsqueeze(0)
a = nib.load(data_meta['training'][0]['label']).get_fdata()

# images = torch.tensor(a).unsqueeze(0)
images = torch.tensor(a).unsqueeze(0)[:, 200:256, 200:250]

def stride_depth_and_inference(model, optimizer, criterion, images_real, patch_size_normal=25, patch_size_low=19, patch_size_out=9, patch_low_factor=3):

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
    images_padded = torch.zeros((batch_size, height_new, width_new, depth_new), dtype=torch.float32)

    # copy the original image to the placeholder
    images_padded[
        :,
        patch_half_low_up: height + patch_half_low_up,
        patch_half_low_up: width + patch_half_low_up,
        patch_half_low_up: depth + patch_half_low_up
    ] = copy.deepcopy(images_real)

    # placeholder to store the inferred/reconstructed image labels
    labels_pred = torch.zeros_like(images_real)

    # indices of the original image
    h_start_orig = 0
    h_end_orig = h_start_orig + patch_size_out

    for index_h in range(patch_half_low_up + patch_half_out, height_new - patch_half_out, patch_size_out):

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

                image_patch_low = images_padded[
                            :,
                            h_start_low: h_end_out,
                            w_start_out: w_end_out,
                            d_start_out: d_end_out
                ]

                patch_out = images_padded[
                            :,
                            h_start_out: h_end_out,
                            w_start_out: w_end_out,
                            d_start_out: d_end_out
                ]

                labels_pred[
                    :,
                    h_start_orig: h_end_orig,
                    w_start_orig: w_end_orig,
                    d_start_orig: d_end_orig
                ] = copy.deepcopy(patch_out)

                d_start_orig = d_start_orig + patch_size_out
                d_end_orig = d_end_orig + patch_size_out

            w_start_orig = w_start_orig + patch_size_out
            w_end_orig = w_end_orig + patch_size_out

        h_start_orig = h_start_orig + patch_size_out
        h_end_orig = h_end_orig + patch_size_out