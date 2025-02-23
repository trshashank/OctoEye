import faery
from pathlib import Path
import octoeye
import scipy.io as sio
import numpy as np
import os


wavelength = 800
integration_time = 5

wavelengths = np.arange(400, 1050, 50)
wavelength_to_index = {wavelength: index for index, wavelength in enumerate(wavelengths)}

mat_file      = "D:/gen4-windows/recordings/event_based_hyperspectral_results.mat"
parent_folder = f"D:/gen4-windows/recordings/{wavelength}"
output_folder = f"D:/Optical_characterisation/event_based_hyperspectral"
if not os.path.exists(f"{output_folder}/{wavelength}"):
    os.makedirs(f"{output_folder}/{wavelength}")
filename = next(Path(f"{parent_folder}").glob("*.es")).stem

data = sio.loadmat(mat_file)
per_event_focal_length = np.loadtxt(f"{parent_folder}/per_event_focal_length.txt", dtype=float)

optimal_focal = data.get('optimal_focal')
segment_range = data.get('segment_range')
optimal_timestamp = data.get('optimal_timestamp')

def get_data_for_wavelength(wavelength):
    if wavelength in wavelength_to_index:
        index = wavelength_to_index[wavelength]
        focal = optimal_focal[0][index]
        segment = segment_range[index]
        timestamp = optimal_timestamp[0][index]
        return focal, segment, timestamp
    else:
        raise ValueError("Wavelength not in the defined range")
focal, segment, timestamp = get_data_for_wavelength(wavelength)


print(f"Wavelength: {wavelength} nm")
print(f"Optimal Focal: {focal}")
print(f"Segment Range: {segment}")
print(f"Optimal Timestamp: {timestamp}")

left_dim, right_dim, top_dim, bottom_dim = octoeye.center_crop_bounds(width=1280, height=720, crop_size=100)

print("Processing cropping")
(
    faery.events_stream_from_file(
        faery.dirname.parent / f"{parent_folder}" / f"{filename}.es",
    )
    .crop(
        left=left_dim,
        right=right_dim,
        top=top_dim,
        bottom=bottom_dim,
    )
    .to_file(
        faery.dirname.parent /  f"{output_folder}" / f"{wavelength}" / f"{wavelength}_{filename}_dvs_crop.es",
    )
)

print("Processing hot pixel filter")
(
    faery.events_stream_from_file(
        faery.dirname.parent / f"{output_folder}" / f"{wavelength}" / f"{wavelength}_{filename}_dvs_crop.es",
    )
    .regularize(frequency_hz=10.0)  # bin events for the hot pixel filter
    .filter_hot_pixels(  # removes pixels significantly more active than their most active neighbor
        maximum_relative_event_count=3,
    )
    .to_file(
        faery.dirname.parent /  f"{output_folder}"/ f"{wavelength}" / f"{wavelength}_{filename}_dvs_without_hot_pixels_crop.es"
    )
)


########################### RENDERING #########################
stream = faery.events_stream_from_file(
    faery.dirname.parent / f"{output_folder}"/ f"{wavelength}" / f"{wavelength}_{filename}_dvs_without_hot_pixels_crop.es"
)

events = stream.to_array()
sensor_size = stream.inner_dimensions


subevents_idx = np.where(np.logical_and(events["t"] >= (timestamp-integration_time)*1e6, 
                                        events["t"] <= (timestamp+integration_time)*1e6))
subevents = events[subevents_idx]


cumulative_map = octoeye.accumulate_py(sensor_size, (0,0), subevents)
rendered_image = octoeye.render(cumulative_map,colormap_name="gray",gamma=lambda image: image ** (1 / 3))
rendered_image.save(f"{output_folder}/{wavelength}_dvs_without_hot_pixels_crop.png")

