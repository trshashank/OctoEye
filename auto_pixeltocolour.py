import argparse
import os
from pathlib import Path
import urllib
import warnings
import numpy as np
from tqdm import tqdm
import octoeye
from PIL import Image
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.filters import threshold_otsu
import imageio
from skimage.morphology import erosion, dilation, opening, closing, white_tophat, disk

folder_path = "/media/samiarja/USB/OctoEye_Dataset/pattern9"
output_folder = "figures"

variance_window_size = 40
dilation_param = 20 #10

def calculate_variance_map(image, roi_size=5, masks=None):
    """
    Calculates the variance map of an image using a sliding window approach.
    Optionally excludes areas specified by one or more binary masks.

    Parameters:
        image (np.ndarray): Input image as a numpy array.
        roi_size (int): Size of the region of interest (ROI).
        masks (list of np.ndarray, optional): List of binary masks where 1 indicates areas to exclude from variance calculation.

    Returns:
        np.ndarray: Variance map of the image.
    """
    height, width = image.shape[:2]
    variance_map = np.zeros((height, width))

    half_roi = roi_size // 2

    # Ensure masks are in a list if not None
    if masks is not None:
        if not isinstance(masks, list):
            masks = [masks]  # Put the single mask into a list
        combined_mask = np.any(masks, axis=0)  # Combine masks logically
    else:
        combined_mask = None

    for i in tqdm(range(half_roi, height - half_roi)):
        for j in range(half_roi, width - half_roi):
            if combined_mask is not None and combined_mask[i, j]:
                continue  # Skip regions specified by the combined mask

            roi = image[i - half_roi:i + half_roi + 1, j - half_roi:j + half_roi + 1]
            variance_map[i, j] = np.var(roi)

    return variance_map

def upscale_image(image, target_size):
    return image.resize(target_size, Image.BILINEAR)

start_time = time.time()
width, height, events = octoeye.read_es_file(f"{folder_path}/event_stream.es")
sensor_size = (width, height)
first_timestamp = events["t"][0]
last_timestamp = events["t"][-1]
end_time = time.time()
colour_pixel_labels = np.zeros(len(events), dtype=int)
print(f"Execution time: {end_time - start_time} seconds")

########################################################################################
red_idx   = np.logical_and(events["t"] >= first_timestamp, events["t"] <= 7.1e6) # [first_timestamp 7.1e6]
green_idx = np.logical_and(events["t"] >= 7.8e6, events["t"] <= 15e6) # [7.8e6 15e6]
blue_idx  = np.logical_and(events["t"] >= 15.3e6, events["t"] <= last_timestamp) # [15.3e6 last_timestamp]
########################################################################################

# feast_weight = octoeye.OctoFEAST_training(events[red_idx])
# print(f"FEAST weight: {feast_weight}")

vx_vel = np.zeros((len(events["x"]), 1)) + 0.0 / 1e6
vy_vel = np.zeros((len(events["y"]), 1)) + 0.0 / 1e6


################ Start processing the red events ################
cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[red_idx],
                                                              colour_pixel_labels[red_idx].astype(np.int32),
                                                              (vx_vel[red_idx], vy_vel[red_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)

variance_map_red = calculate_variance_map(cumulative_map_object.pixels, roi_size=variance_window_size, masks=None)

# plt.imshow(variance_map_red, cmap='hot')
# plt.savefig('figures/0_variance_map.png')

variance_map_red_cp = variance_map_red.copy()
thresh = threshold_otsu(variance_map_red_cp)
variance_map_red_cp = np.where(variance_map_red_cp > thresh * 0.5, 1, 0)


upscaled_variance_map_red = upscale_image(Image.fromarray(variance_map_red), warped_image_segmentation_rgb.size)
red_binary_mask = upscale_image(Image.fromarray((variance_map_red_cp * 255).astype(np.uint8)), warped_image_segmentation_rgb.size)
red_binary_mask_array = np.array(red_binary_mask, dtype=bool)

red_binary_mask.save('figures/0_variance_map.png')

# make a copy of the red mask but with dilation
red_binary_mask_cp = red_binary_mask_array.copy()
dilated_red_mask = dilation(red_binary_mask_cp, disk(dilation_param))


colour_pixel_labels = octoeye.colour_labeling(events[red_idx], 
                                              colour_pixel_labels, 
                                              red_binary_mask_array, 
                                              1,
                                              red_idx, 
                                              colour='Red')


cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[red_idx],
                                                              colour_pixel_labels[red_idx].astype(np.int32),
                                                              (vx_vel[red_idx], vy_vel[red_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)
warped_image_segmentation_rgb.save("figures/0_red_events.png")


################ Start processing the green events ################
cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[green_idx],
                                                              colour_pixel_labels[green_idx].astype(np.int32),
                                                              (vx_vel[green_idx], vy_vel[green_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)

variance_map_green = calculate_variance_map(cumulative_map_object.pixels, roi_size=variance_window_size, masks=[dilated_red_mask])

# plt.imshow(variance_map_green, cmap='hot')
# plt.savefig('figures/1_variance_map.png')

variance_map_green_cp = variance_map_green.copy()
thresh = threshold_otsu(variance_map_green_cp)
variance_map_green_cp = np.where(variance_map_green_cp > thresh * 0.5, 1, 0)


upscaled_variance_map_green = upscale_image(Image.fromarray(variance_map_green), warped_image_segmentation_rgb.size)
green_binary_mask = upscale_image(Image.fromarray((variance_map_green_cp * 255).astype(np.uint8)), warped_image_segmentation_rgb.size)
green_binary_mask_array = np.array(green_binary_mask, dtype=bool)

green_binary_mask.save('figures/1_variance_map.png')

# make a copy of the red mask but with dilation
green_binary_mask_cp = green_binary_mask_array.copy()
dilated_green_mask = dilation(green_binary_mask_cp, disk(dilation_param))


colour_pixel_labels = octoeye.colour_labeling(events[green_idx], 
                                              colour_pixel_labels, 
                                              green_binary_mask_array, 
                                              1,
                                              green_idx, 
                                              colour='Green')

cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[green_idx],
                                                              colour_pixel_labels[green_idx].astype(np.int32),
                                                              (vx_vel[green_idx], vy_vel[green_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)
warped_image_segmentation_rgb.save("figures/1_green_events.png")

################ Start processing the blue events ################
cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[blue_idx],
                                                              colour_pixel_labels[blue_idx].astype(np.int32),
                                                              (vx_vel[blue_idx], vy_vel[blue_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)

variance_map_blue = calculate_variance_map(cumulative_map_object.pixels, roi_size=variance_window_size, masks=[dilated_red_mask, dilated_green_mask])

# plt.imshow(variance_map_blue, cmap='hot')
# plt.savefig('figures/2_variance_map.png')

variance_map_blue_cp = variance_map_blue.copy()
thresh = threshold_otsu(variance_map_blue_cp)
variance_map_blue_cp = np.where(variance_map_blue_cp > thresh * 0.5, 1, 0)

upscaled_variance_map_blue = upscale_image(Image.fromarray(variance_map_blue), warped_image_segmentation_rgb.size)
blue_binary_mask = upscale_image(Image.fromarray((variance_map_blue_cp * 255).astype(np.uint8)), warped_image_segmentation_rgb.size)
blue_binary_mask_array = np.array(blue_binary_mask, dtype=bool)

blue_binary_mask.save('figures/2_variance_map.png')

# make a copy of the blue mask but with dilation
blue_binary_mask_cp = blue_binary_mask_array.copy()
dilated_blue_mask = dilation(blue_binary_mask_cp, disk(dilation_param))


colour_pixel_labels = octoeye.colour_labeling(events[blue_idx], 
                                              colour_pixel_labels, 
                                              blue_binary_mask_array, 
                                              1,
                                              blue_idx, 
                                              colour='Blue')

cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height),
                                                              events[blue_idx],
                                                              colour_pixel_labels[blue_idx].astype(np.int32),
                                                              (vx_vel[blue_idx], vy_vel[blue_idx]))
warped_image_segmentation_rgb = octoeye.rgb_render(cumulative_map_object, seg_label)
warped_image_segmentation_rgb.save("figures/2_blue_events.png")

print("Saving the labels")
# Convert colour_pixel_labels to uint8 to reduce memory usage
colour_pixel_labels = colour_pixel_labels.astype(np.uint8)

# Save the labels as a compressed numpy file
np.savez_compressed(f"{folder_path}/colour_pixel_labels.npz", colour_pixel_labels=colour_pixel_labels)