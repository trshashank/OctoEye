import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve

import packages.conversion.format
import packages.conversion.h5writer
from collections import namedtuple

'''
python3 python/convert.py --input_file /media/samiarja/USB/raw/recording_2024-12-07_16-13-36_pattern9_backup.raw
'''

def CrossConv(input_events, ratio_par):
    print("Start removing hot pixels...")

    # Check if the input events are empty
    if len(input_events.x) == 0 or len(input_events.y) == 0:
        print("Input events are empty. Skipping processing.")
        Events = namedtuple('Events', ['x', 'y', 'p', 't', 'size'])
        # Return empty output events with the same structure
        output_events = Events(
            x=np.array([], dtype=np.uint16),
            y=np.array([], dtype=np.uint16),
            p=np.array([], dtype=np.uint8),
            t=np.array([], dtype=np.int64),
            size=0
        )
        detected_noise = np.array([], dtype=int)  # Empty detected noise array
        return output_events, detected_noise

    x = input_events.x
    y = input_events.y
    x_max, y_max = x.max() + 1, y.max() + 1

    # Create a 2D histogram of event counts
    event_count = np.zeros((x_max, y_max), dtype=int)
    np.add.at(event_count, (x, y), 1)

    kernels = [
        np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    ]

    # Apply convolution with each kernel and calculate ratios
    shifted = [convolve(event_count, k, mode="constant", cval=0.0) for k in kernels]
    max_shifted = np.maximum.reduce(shifted)
    ratios = event_count / (max_shifted + 1.0)
    smart_mask = ratios < ratio_par  # this should be 2 ideally

    yhot, xhot = np.where(~smart_mask)
    label_hotpix = np.zeros(len(input_events.x), dtype=bool)
    for xi, yi in zip(xhot, yhot):
        label_hotpix |= (y == xi) & (x == yi)
    label_hotpix_binary = label_hotpix.astype(int)

    jj = np.where(label_hotpix_binary == 1)[0]
    mask = np.ones(input_events.x.shape[0], dtype=bool)  # Start with all True
    mask[jj] = False  # Set the indices in `jj` to False

    Events = namedtuple('Events', ['x', 'y', 'p', 't', 'size'])

    # Filter events using the mask
    output_events = Events(
        x=input_events.x[mask],
        y=input_events.y[mask],
        p=input_events.p[mask],
        t=input_events.t[mask],
        size=np.sum(mask)  # Update size to reflect the new number of events
    )

    print(f'Number of input event: {len(input_events.y)}')
    print(f'Number of hot pixel events: {np.sum(label_hotpix)}')

    detected_noise = label_hotpix_binary
    detected_noise = 1 - detected_noise
    return output_events, detected_noise



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert events to h5 format to prepare for calibration.')
    parser.add_argument('--input_file', default="", help='Path to file which will be converted to h5 format.')
    parser.add_argument('--output_file', '-o', default="", help='Output path for h5 file. Default: Input path but with h5 suffix.')
    parser.add_argument('--ros_topic', '-rt', default='/dvs/events', help='ROS topic for events if input file is a rosbag.')

    args = parser.parse_args()

    input_file = Path(args.input_file)
    assert input_file.exists()
    if args.output_file:
        output_file = Path(args.output_file)
        assert output_file.suffix == '.h5'
    else:
        output_file = Path(input_file).parent / (input_file.stem + '.h5')
    assert not output_file.exists(), "{} already exists.".format(output_file)

    rostopic = args.ros_topic

    event_generator = packages.conversion.format.get_generator(input_file, delta_t_ms=100, topic=rostopic)
    h5writer = packages.conversion.h5writer.H5Writer(output_file)

    counter = 0
    for events in event_generator():
        ## filter event before writing them to h5
        output_events, detected_noise = CrossConv(events, 2)
        
        ## save this chunk of events into a .txt file
        with open(f"/media/samiarja/USB/OctoEye_Dataset/pattern10/txt_per_frame/{counter}.txt", "w") as file:
            # Iterate through each event
            for i in range(len(output_events.t)):
                # Write formatted string to the file
                file.write(f"{output_events.t[i]} {output_events.x[i]} {output_events.y[i]} {output_events.p[i]}\n")
                
        counter += 1
        
        h5writer.add_data(output_events)
