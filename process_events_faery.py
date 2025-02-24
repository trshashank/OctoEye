import faery
from pathlib import Path
import octoeye

wavelength    = 400
parent_folder = f"D:/gen4-windows/recordings/{wavelength}"
filename = next(Path(f"{parent_folder}").glob("*.es")).stem


########################### EVENT CROPPING #########################
print("Processing cropping")
left_dim, right_dim, top_dim, bottom_dim = octoeye.center_crop_bounds(width=1280, height=720, crop_size=500)
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
        faery.dirname.parent /  f"{parent_folder}" / f"{filename}_dvs_crop.es",
    )
)


########################### EVENT HOT PIXEL FILTER #########################
print("Processing hot pixel filter")
(
    faery.events_stream_from_file(
        faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_crop.es",
    )
    .regularize(frequency_hz=10.0)  # bin events for the hot pixel filter
    .filter_hot_pixels(  # removes pixels significantly more active than their most active neighbor
        maximum_relative_event_count=3.0,
    )
    .to_file(
        faery.dirname.parent /  f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
    )
)


########################### EVENT RATE #########################
print("Processing event rate")
event_rate = faery.events_stream_from_file(
    faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
).to_event_rate()

# event_rate = faery.events_stream_from_file(
#     faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
# ).remove_off_events().to_event_rate()

event_rate.to_file(
    faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_event_rate.png",
)


########################### WIGGLE PLOT #########################
print("Processing wiggle plot")
stream = faery.events_stream_from_file(
    faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
)

wiggle_parameters = faery.WiggleParameters(time_range=stream.time_range())

(
    stream.regularize(frequency_hz=wiggle_parameters.frequency_hz)
    .render(
        decay=wiggle_parameters.decay,
        tau=wiggle_parameters.tau,
        colormap=faery.colormaps.managua.flipped(),
    )
    .scale()
    .add_timecode(output_frame_rate=wiggle_parameters.frame_rate)
    .to_file(
        path=faery.dirname.parent / f"{parent_folder}" / f"{filename}_wiggle.gif",
        frame_rate=wiggle_parameters.frame_rate,
        rewind=wiggle_parameters.rewind,
        skip=wiggle_parameters.skip,
        on_progress=faery.progress_bar,
    )
)


############################# EVENT FRAME #########################
print("Processing event frame")

(
    faery.events_stream_from_file(
        faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
    )
    .regularize(frequency_hz=1)
    .render(
        decay="exponential",
        tau="00:00:00.200000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .to_files(
        faery.dirname.parent 
        / f"{parent_folder}" 
        / "dvs_frames"
        / "{index:04}.png",
    )
)


########################### EVENT VIDEO #########################
print("Processing event video")
(
    faery.events_stream_from_file(
        faery.dirname.parent / f"{parent_folder}" / f"{filename}_dvs_without_hot_pixels_crop.es"
    )
    .regularize(frequency_hz=1)
    .render(
        decay="cumulative",
        tau="00:00:00.500000",
        colormap=faery.colormaps.managua.flipped(),
    )
    .scale()
    .add_timecode()
    .to_file(faery.dirname.parent / f"{parent_folder}" / f"{filename}_video.mp4")
)
