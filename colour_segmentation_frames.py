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
import subprocess

folder_path = "/media/samiarja/USB/OctoEye_Dataset/pattern9"

start_time = time.time()
width, height, events = octoeye.read_es_file(f"{folder_path}/event_stream.es")
sensor_size = (width, height)
last_timestamp = events["t"][-1]
vx_vel = np.zeros((len(events["x"]), 1)) + 0.0 / 1e6
vy_vel = np.zeros((len(events["y"]), 1)) + 0.0 / 1e6
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# load the .npz file
data = np.load(f"{folder_path}/colour_pixel_labels.npz")
labels = data["colour_pixel_labels"]

print("Start saving the frames")
sequence_number = 0
chunk_timediff = 5e4

for chunk_start in np.arange(0, last_timestamp + chunk_timediff, chunk_timediff):
    chunk_end = chunk_start + chunk_timediff
    ii = np.where(np.logical_and(events["t"] >= chunk_start, events["t"] < chunk_end))
    cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb(
        (width, height),
        events[ii],
        labels[ii].astype(np.int32),
        (vx_vel[ii], vy_vel[ii])
    )
    warped_image_segmentation_rgb = octoeye.rgb_render_white_octoeye(cumulative_map_object, seg_label)
    output_filename = f"{folder_path}/colour_segmentation_frames/{sequence_number:06d}.png"
    # flipped_warped_image_segmentation_rgb = warped_image_segmentation_rgb.transpose(Image.FLIP_TOP_BOTTOM)
    warped_image_segmentation_rgb.save(output_filename)
    sequence_number += 1
    print(f"Saved frame {output_filename}")
    

print("All frames saved. Starting video generation.")
# FFmpeg Command to Generate Video
output_video_path = f"{folder_path}/colour_segmentation_frames/colour_segmentation_video.mp4"
frame_rate = 30  # User can modify this value

ffmpeg_command = [
    "ffmpeg",
    "-y",  # Overwrite output file if it exists
    "-framerate", str(frame_rate),  # Set frame rate
    "-i", f"{folder_path}/colour_segmentation_frames/%06d.png",  # Input file pattern
    "-c:v", "libx264",  # Use H.264 codec
    "-preset", "fast",  # Encoding preset for speed
    "-crf", "18",  # Quality (lower is better, 18-28 is a good range)
    "-pix_fmt", "yuv420p",  # Pixel format for compatibility
    output_video_path
]

# Run FFmpeg command
subprocess.run(ffmpeg_command, check=True)

print(f"Video saved at {output_video_path}")