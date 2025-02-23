import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import octoeye
import faery
import re
from scipy.io import savemat

# Initialize global variable for enforcing monotonicity across wavelengths.
prev_optimal_focal = 0.0

# This list will hold one record per wavelength.
results_list = []

# Define the plot range (in seconds) within which we consider candidates.
PLOT_RANGE = (200, 600)  # Adjust as needed

for wv in range(400, 1050, 50):
    wavelength = wv
    parent_directory = f"D:/gen4-windows/recordings/{wavelength}"
    event_filename = next(Path(parent_directory).glob("*_dvs_without_hot_pixels_crop.es")).stem
    gen4_triggers_path = next(Path(parent_directory).glob("*.jsonl")).stem
    teensy_log_path = f"{parent_directory}/motor_triggers.txt"

    # Load event stream and sensor dimensions.
    stream = faery.events_stream_from_file(
        faery.dirname.parent / parent_directory / f"{event_filename}.es"
    )
    events = stream.to_array()
    sensor_size = stream.inner_dimensions  # type: ignore

    # Compute event rate (using off events removed).
    event_rate = faery.events_stream_from_file(
        faery.dirname.parent / parent_directory / f"{event_filename}.es"
    ).remove_off_events().to_event_rate()

    # ------------------------------
    # PARAMETERS FOR DYNAMIC CONVERSION
    # ------------------------------
    feedback_min = 284     # raw feedback when camera is farthest
    feedback_max = 3965    # raw feedback when camera is closest
    distance_at_min_mm = 92  # in mm (i.e. 92 cm)
    feedback_range = feedback_max - feedback_min

    # ------------------------------
    # LOAD TEENSY DATA (/motor_triggers.txt)
    # ------------------------------
    # Assumes two columns: raw_timestamp (ns) and feedback (raw value)
    teensy_data = np.loadtxt(teensy_log_path, delimiter=",", dtype=int)
    unix_motor_timestamps = teensy_data[:, 0]  # in nanoseconds
    motor_feedback_values = teensy_data[:, 1]
    
    # ------------------------------
    # PARSE GEN4 TRIGGERS (EVENT CAMERA)
    # ------------------------------
    rising_events = []
    falling_events = []
    start_recording_info = {}

    def fix_invalid_escapes(s):
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    with open(f"{parent_directory}/{gen4_triggers_path}.jsonl", "r") as file:
        for line in file:
            fixed_line = fix_invalid_escapes(line.strip())
            data = json.loads(fixed_line)
            if data["type"] == "start_recording":
                start_recording_info = {
                    "filename": data["payload"]["filename"],
                    "initial_t": data["payload"]["initial_t"]
                }
            elif data["type"] == "trigger_event":
                event_payload = data["payload"]
                event_info = {
                    "t": event_payload["t"],
                    "system_timestamp": event_payload["system_timestamp"],
                    "id": event_payload["id"]
                }
                if event_payload["rising"]:
                    rising_events.append(event_info)
                else:
                    falling_events.append(event_info)

    initial_t = start_recording_info["initial_t"]
    # Adjust events timestamps by the initial recording time.
    events["t"] = events["t"] + initial_t

    # ------------------------------
    # OFFSET CORRECTION & TIMESTAMP EXTRACTION
    # ------------------------------
    offset = unix_motor_timestamps[0] - rising_events[0]['system_timestamp']
    events_timestamp = [event["t"] for event in rising_events]
    events_unix_timestamp = [event["system_timestamp"] for event in rising_events]
    offset_corrected_motor_timestamps = unix_motor_timestamps - offset

    # ------------------------------
    # CONVERT FEEDBACK TO FOCAL LENGTH (in mm then to cm)
    # ------------------------------
    # For each motor trigger feedback value, convert to focal length (cm)
    focal_length_in_mm = (feedback_max - motor_feedback_values) / feedback_range * distance_at_min_mm
    focal_lengths_cm = focal_length_in_mm / 10  # convert mm to cm

    # ------------------------------
    # INTERPOLATE FOCAL LENGTH PER EVENT
    # ------------------------------
    focal_length_per_events = np.empty(len(events["x"]))
    focal_length_per_events[:] = np.nan
    for idx in range(len(unix_motor_timestamps) - 1):
        first_timestamp = events_timestamp[idx]
        second_timestamp = events_timestamp[idx + 1]
        first_focal = focal_lengths_cm[idx]
        second_focal = focal_lengths_cm[idx + 1]
        seg_indices = np.where((events["t"] >= first_timestamp) & (events["t"] <= second_timestamp))[0]
        if seg_indices.size == 0:
            continue
        sub_t = events["t"][seg_indices]
        fraction = (sub_t - first_timestamp) / (second_timestamp - first_timestamp)
        focal_length_per_events[seg_indices] = first_focal + fraction * (second_focal - first_focal)

    # Convert events and event_rate timestamps to seconds.
    events_seconds = events["t"] / 1e6  
    event_rate_seconds = (event_rate.timestamps + initial_t) / 1e6

    # ------------------------------
    # CANDIDATE EXTRACTION PER SEGMENT
    # ------------------------------
    # For each motor trigger segment (indices 0 to N-2), extract:
    #   - the candidate optimal focal value (via interpolation at the local event rate peak)
    #   - the candidate timestamp (in seconds)
    #   - the segment range: motor trigger endpoints (focal lengths in cm and event timestamps in seconds)
    #   - the event rate array (timestamps and values) for that segment.
    candidates = []
    num_segments = len(unix_motor_timestamps) - 1
    for idx in range(num_segments):
        seg_start_event = events_timestamp[idx]
        seg_end_event = events_timestamp[idx + 1]
        # Get indices for event_rate within this segment.
        mask = ((event_rate.timestamps + initial_t) >= seg_start_event) & ((event_rate.timestamps + initial_t) <= seg_end_event)
        seg_event_indices = np.where(mask)[0]
        if seg_event_indices.size == 0:
            continue

        # Find the local peak (maximum event rate) within this segment.
        local_peak_idx = seg_event_indices[np.argmax(event_rate.samples[seg_event_indices])]
        candidate_timestamp = (event_rate.timestamps[local_peak_idx] + initial_t) / 1e6
        # Only consider candidates within the defined PLOT_RANGE.
        if candidate_timestamp < PLOT_RANGE[0] or candidate_timestamp > PLOT_RANGE[1]:
            continue

        # Interpolate the focal length at the candidate timestamp.
        candidate_focal = np.interp(candidate_timestamp, events_seconds, focal_length_per_events)
        
        # Define the motor-trigger segment range.
        seg_start_focal = focal_lengths_cm[idx]
        seg_end_focal = focal_lengths_cm[idx + 1]
        seg_start_time = seg_start_event / 1e6
        seg_end_time = seg_end_event / 1e6
        segment_range = ((seg_start_focal, seg_end_focal), (seg_start_time, seg_end_time))
        
        # Extract the full event rate array for this segment.
        seg_event_rate_timestamps = (event_rate.timestamps[seg_event_indices] + initial_t) / 1e6
        seg_event_rate_values = event_rate.samples[seg_event_indices]
        event_rate_array = {
            'timestamps': seg_event_rate_timestamps,
            'values': seg_event_rate_values
        }
        
        candidate = {
            'optimal_focal': candidate_focal,
            'optimal_timestamp': candidate_timestamp,
            'segment_range': segment_range,
            'event_rate_array': event_rate_array,
        }
        candidates.append(candidate)

    # ------------------------------
    # SELECT THE OPTIMAL CANDIDATE FOR THIS WAVELENGTH
    # ------------------------------
    if candidates:
        valid_candidates = [cand for cand in candidates if cand['optimal_focal'] > prev_optimal_focal]
        if valid_candidates:
            selected_candidate = min(valid_candidates, key=lambda c: c['optimal_focal'])
        else:
            selected_candidate = min(candidates, key=lambda c: c['optimal_focal'])
        prev_optimal_focal = selected_candidate['optimal_focal']
    else:
        selected_candidate = {
            'optimal_focal': np.nan,
            'optimal_timestamp': np.nan,
            'segment_range': ((np.nan, np.nan), (np.nan, np.nan)),
            'event_rate_array': {'timestamps': np.array([]), 'values': np.array([])},
        }

    # ------------------------------
    # EXTRACTION OF EVENTS FROM THE SELECTED SEGMENT
    # ------------------------------
    # Using the candidate's segment_range timestamps, select the events (x, y, p, t, focal) within that range.
    (seg_focal_range, seg_time_range) = selected_candidate['segment_range']
    seg_start_time, seg_end_time = seg_time_range
    mask_events = (events_seconds >= seg_start_time) & (events_seconds <= seg_end_time)
    selected_events = {
        'x': events["x"][mask_events],
        'y': events["y"][mask_events],
        'p': events["p"][mask_events],
        't': events_seconds[mask_events],
        'focal': focal_length_per_events[mask_events]
    }

    # ------------------------------
    # (Optional) Plot the results for this wavelength.
    # ------------------------------
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(events_seconds, focal_length_per_events, label="Focal Length", color="red", lw=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Focal Length (cm)", color="red")
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()
    ax2.plot(event_rate_seconds, event_rate.samples, label="Event Rate", color="blue", lw=2)
    ax2.set_ylabel("Event Rate (events/s)", color="blue")
    ax2.tick_params(axis='y', labelcolor='blue')
    ax1.set_xlim(PLOT_RANGE)
    
    if candidates:
        for cand in candidates:
            ax1.axvline(x=cand['optimal_timestamp'], color='lightgreen', linestyle='--', lw=1)
        ax1.axvline(x=selected_candidate['optimal_timestamp'], color='green', linestyle='--', lw=2)
        ax1.plot(selected_candidate['optimal_timestamp'], selected_candidate['optimal_focal'],
                 'o', color='green', markersize=8)
        y_top = ax1.get_ylim()[1] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        ax1.text(selected_candidate['optimal_timestamp'], y_top, f"{selected_candidate['optimal_focal']:.4f}",
                 color='green', fontsize=10, ha="center", va="top", backgroundcolor="w", rotation=90)
    # plt.show()  # Uncomment to display the plot if needed.

    # ------------------------------
    # STORE THE RECORD FOR THIS WAVELENGTH
    # ------------------------------
    # Each record now includes:
    #   1. The wavelength.
    #   2. The optimal focal point (in cm).
    #   3. The optimal timestamp (s).
    #   4. The segment range ((start_focal, end_focal), (start_time, end_time)).
    #   5. The event rate array for that segment.
    #   6. The events (x, y, p, t, focal) from the selected segment.
    record = (
        wavelength,
        selected_candidate['optimal_focal'],
        selected_candidate['optimal_timestamp'],
        selected_candidate['segment_range'],
        selected_candidate['event_rate_array'],
        selected_events
    )
    results_list.append(record)

# ------------------------------
# CREATE A NUMPY STRUCTURED ARRAY TO STORE ALL RESULTS
# ------------------------------
dtype = np.dtype([
    ('wavelength', np.int32),
    ('optimal_focal', np.float64),
    ('optimal_timestamp', np.float64),
    ('segment_range', object),          # ((start_focal, end_focal), (start_time, end_time))
    ('event_rate_array', object),         # dict with keys 'timestamps' and 'values'
    ('events', object)                   # dict with keys 'x', 'y', 'p', 't', 'focal'
])
structured_results = np.array(results_list, dtype=dtype)

print(structured_results)

# Convert to a dictionary for MATLAB.
mat_data = {}
for field in structured_results.dtype.names:
    if structured_results.dtype[field].kind == 'O':
        mat_data[field] = structured_results[field].tolist()
    else:
        mat_data[field] = structured_results[field]

# Save the dictionary to a .mat file.
savemat(f"D:/gen4-windows/recordings/event_based_hyperspectral_results.mat", mat_data)


sorted_idx = np.argsort(structured_results['optimal_focal'])
sorted_results = structured_results[sorted_idx]

plt.figure(figsize=(10, 6))
for rec in sorted_results:
    # Extract the segment range and event rate array from the record.
    # segment_range is a tuple: ((seg_start_focal, seg_end_focal), (seg_start_time, seg_end_time))
    seg_range = rec['segment_range']
    (seg_start_focal, seg_end_focal), (seg_start_time, seg_end_time) = seg_range
    er_array = rec['event_rate_array']
    t_er = er_array['timestamps']  # event rate timestamps (in seconds)
    values_er = er_array['values']   # corresponding event rate values

    # Map the event rate timestamps to focal length using linear interpolation
    focal_vals = np.interp(t_er, [seg_start_time, seg_end_time], [seg_start_focal, seg_end_focal])
    
    # Plot the event rate curve for this record
    plt.plot(focal_vals, values_er, label=f"{rec['wavelength']} nm (f={rec['optimal_focal']:.2f} cm)")

plt.xlabel("Focal Length (cm)")
plt.ylabel("Event Rate (events/s)")
plt.title("Event Rate Range Sorted by Focal Length")
plt.legend()
plt.show()
print("Done")