import os
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import subprocess
from tqdm import tqdm
import time
import octoeye
from skimage.filters import threshold_otsu

parent_path = "/media/samiarja/USB/OctoEye_Dataset/pattern9"
E2VID_freq  = 5

frames_dir = f"{parent_path}/greylevel_reconstructed_frames"
output_dir = f"{parent_path}/RGB_reconstructed_frames"
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
width, height, events = octoeye.read_es_file(f"{parent_path}/event_stream.es")
data = np.load(f"{parent_path}/colour_pixel_labels.npz")
labels = data["colour_pixel_labels"]
end_time = time.time()
sensor_size = (width, height)
first_timestamp = events["t"][0]
last_timestamp = events["t"][-1]
colour_pixel_labels = np.zeros(len(events), dtype=int)
vx_vel = np.zeros((len(events["x"]), 1)) + 0.0 / 1e6
vy_vel = np.zeros((len(events["y"]), 1)) + 0.0 / 1e6
print(f"Execution time: {end_time - start_time} seconds")


# Combined settings for colors, time bounds, and labels
color_settings = {
    "Red": {"focal_window": (first_timestamp, 7.1e6), "label": 1},
    "Green": {"focal_window": (7.8e6, 15e6), "label": 2},
    "Blue": {"focal_window": (15.3e6, last_timestamp), "label": 3},
}

# Prepare to process frames
E2VID_frames = sorted(os.listdir(frames_dir))
sequence_number = 0
chunk_timediff = (1 / E2VID_freq) * 1e6  # Convert frequency to microseconds per frame

# Process each color segment
for color, settings in color_settings.items():
    start_time, end_time = settings["focal_window"]
    label = settings["label"]

    # Ensure accurate handling of each time chunk within the color segment
    for chunk_start in np.arange(start_time, end_time, chunk_timediff):
        chunk_end = min(chunk_start + chunk_timediff, end_time)  # Avoid exceeding the segment's end time

        # Select events matching the time range and label condition
        ii = np.where((events["t"] >= chunk_start) & (events["t"] < chunk_end) & (labels == label))[0]
        
        frame_path = os.path.join(frames_dir, E2VID_frames[sequence_number])
        frame = Image.open(frame_path)
        
        # Process events to generate segmentation and apply overlay
        cumulative_map_object, seg_label = octoeye.accumulate_cnt_rgb((width, height), events[ii], labels[ii].astype(np.int32), (vx_vel[ii], vy_vel[ii]))
        threshold_val = threshold_otsu(cumulative_map_object.pixels)
        cumulative_map_object.pixels[cumulative_map_object.pixels < threshold_val] = 0
        warped_image_segmentation_rgb = octoeye.rgb_render_white_octoeye(cumulative_map_object, seg_label)
        
        binary_frame_from_events = np.array(warped_image_segmentation_rgb.convert('L'))
        binary_frame_from_events[binary_frame_from_events<255]=1
        binary_frame_from_events[binary_frame_from_events==255]=0
        
        overlayed_frame = octoeye.apply_blurry_overlay(frame, binary_frame_from_events, color, sigma=3)
        
        output_frame_path = os.path.join(output_dir, E2VID_frames[sequence_number % len(E2VID_frames)])
        overlayed_frame.save(output_frame_path)
        sequence_number += 1
        print(f"Saved frame {output_frame_path} with color {color}")
            
        if chunk_end == end_time:
            break  # Stop the loop if the end of the current color segment is reached
        


print("All frames saved. Starting video generation.")
# FFmpeg Command to Generate Video
output_video_path = f"{output_dir}/colour_segmentation_video.mp4"
frame_rate = 30  # User can modify this value

ffmpeg_command = [
    "ffmpeg",
    "-y",  # Overwrite output file if it exists
    "-framerate", str(frame_rate),  # Set frame rate
    "-i", f"{output_dir}/%05d.png",  # Input file pattern
    "-c:v", "libx264",  # Use H.264 codec
    "-preset", "fast",  # Encoding preset for speed
    "-crf", "18",  # Quality (lower is better, 18-28 is a good range)
    "-pix_fmt", "yuv420p",  # Pixel format for compatibility
    output_video_path
]

# Run FFmpeg command
subprocess.run(ffmpeg_command, check=True)
print(f"Video saved at {output_video_path}")