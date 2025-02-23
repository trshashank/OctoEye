import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import octoeye
import faery
import re


wavelength                      = 1000
PLOT_RANGE                      = (450,650) #(700,900)
parent_directory                = f"D:/gen4-windows/recordings/{wavelength}"
event_filename                  = next(Path(f"{parent_directory}").glob("*_dvs_without_hot_pixels_crop.es")).stem
gen4_triggers_path              = next(Path(f"{parent_directory}").glob("*.jsonl")).stem
teensy_log_path                 = f"{parent_directory}/motor_triggers.txt"

stream = faery.events_stream_from_file(
    faery.dirname.parent / f"{parent_directory}" / f"{event_filename}.es"
)

events = stream.to_array()
sensor_size = stream.inner_dimensions

event_rate = faery.events_stream_from_file(
    faery.dirname.parent / f"{parent_directory}" / f"{event_filename}.es"
).to_event_rate()

# event_rate = faery.events_stream_from_file(
#     faery.dirname.parent / f"{parent_directory}" / f"{event_filename}.es"
# ).remove_off_events().to_event_rate()


# ------------------------------
# PARAMETERS FOR DYNAMIC CONVERSION
# ------------------------------
feedback_min = 284     # At this raw feedback, camera is farthest away.
feedback_max = 3965    # At this raw feedback, camera is closest.
distance_at_min_mm = 92  # When feedback is at minimum, camera is 920 mm (92 cm) away from the ball lens.
feedback_range = feedback_max - feedback_min

# ------------------------------
# LOAD TEENSY DATA
# ------------------------------
# Assuming the Teensy log file has two columns: raw_timestamp (ns) and feedback (raw value)
teensy_data = np.loadtxt(teensy_log_path, delimiter=",", dtype=int)
unix_motor_timestamps = teensy_data[:, 0]   # in nanoseconds
motor_feedback_values = teensy_data[:, 1]

# ------------------------------
# PARSE GEN4 TRIGGERS (EVENT CAMERA)
# ------------------------------
rising_events = []
falling_events = []
start_recording_info = {}

def fix_invalid_escapes(s):
    # Replace any backslash that is not followed by a valid JSON escape character
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
events["t"] = events["t"] + initial_t
# ------------------------------
# COMPUTE OFFSET CORRECTION
# ------------------------------
# We assume that the first Teensy event and the first Gen4 rising event correspond to the same moment.
offset = unix_motor_timestamps[0] - rising_events[0]['system_timestamp']

events_timestamp = [event["t"] for event in rising_events]
events_unix_timestamp = [event["system_timestamp"] for event in rising_events]

# ------------------------------
# COMPUTE CORRECTED TIMESTAMPS
# ------------------------------
# Apply offset correction and convert from nanoseconds to seconds.
offset_corrected_motor_timestamps = unix_motor_timestamps - offset

# ------------------------------
# CONVERT FEEDBACK TO FOCAL LENGTH (in mm)
# ------------------------------
# Dynamic conversion: as the raw feedback increases from 284 to 3965, the camera moves from 920 mm away to 0 mm.
focal_length_in_mm = (feedback_max - motor_feedback_values) / feedback_range * distance_at_min_mm

# Save new log file with four columns: raw_timestamp, feedback, corrected_timestamp_sec, focal_length_in_mm
data_with_focal_length = np.column_stack((unix_motor_timestamps, 
                                          offset_corrected_motor_timestamps,
                                          events_unix_timestamp,
                                          events_timestamp,
                                          motor_feedback_values,  
                                          focal_length_in_mm/10))

header_line_focal = "unix_motor_timestamps,offset_corrected_motor_timestamps,events_unix_timestamp,events_timestamp,motor_feedback_values,focal_length_in_cm"
np.savetxt(f"{parent_directory}/aligned_timestamps.txt", data_with_focal_length, delimiter=",", fmt="%d,%d,%d,%d,%d,%.6f",
           header=header_line_focal, comments="")

# ------------------------------
# PLOT FOCAL LENGTH VS. CORRECTED TIMESTAMP
# ------------------------------
##### label the events with the focal length using the correct timstamp

# --- Compute Focal Length Per Event via Linear Interpolation ---
focal_length_per_events = np.empty(len(events["x"]))
focal_length_per_events[:] = np.nan
for idx in range(len(unix_motor_timestamps) - 1):
    first_timestamp = events_timestamp[idx]
    second_timestamp = events_timestamp[idx + 1]
    
    first_focal_len = focal_length_in_mm[idx] / 10
    second_focal_len = focal_length_in_mm[idx + 1] / 10

    event_idx = np.where((events["t"] >= first_timestamp) & (events["t"] <= second_timestamp))[0]
    if event_idx.size == 0:
        continue

    sub_t = events["t"][event_idx]
    fraction = (sub_t - first_timestamp) / (second_timestamp - first_timestamp)
    
    # Interpolate focal length for these events.
    focal_length_per_events[event_idx] = first_focal_len + fraction * (second_focal_len - first_focal_len)
    # Optionally, print time diff info:
    # print(f"time diff {idx}: {(events['t'][event_idx][-1] - events['t'][event_idx][0])/1e6} s, Events: {len(event_idx)}")

np.savetxt(f"{parent_directory}/per_event_focal_length.txt", focal_length_per_events)

# --- Convert Timestamps to Seconds ---
events_seconds = events["t"] / 1e6  
event_rate_seconds = (event_rate.timestamps + initial_t) / 1e6

# --- Create Plot ---
fig, ax1 = plt.subplots(figsize=(12, 6))
# Plot focal length on the left y-axis.
ax1.plot(events_seconds, focal_length_per_events, label="Focal Length", color="red", lw=2)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Focal Length (cm)", color="red")
ax1.tick_params(axis='y', labelcolor='red')

# Create a second y-axis for the event rate samples.
ax2 = ax1.twinx()
ax2.plot(event_rate_seconds, event_rate.samples, label="Event Rate", color="blue", lw=2)
ax2.set_ylabel("Event Rate (events/s)", color="blue")
ax2.tick_params(axis='y', labelcolor='blue')

# Limit the x-axis to a specific window (here, from 400 to 600 s).
ax1.set_xlim(PLOT_RANGE)
# ax2.set_ylim((0,1000))


focal_points = []
peak_times_array = []
# --- For Each Focal Range: Find Maximum Event Rate and Annotate ---
# Loop through each focal range defined by the motor timestamps.
for idx in range(len(unix_motor_timestamps) - 1):
    first_timestamp = events_timestamp[idx]
    second_timestamp = events_timestamp[idx + 1]
    
    # Create a mask for event rate times within this focal range.
    mask = ((event_rate.timestamps + initial_t) >= first_timestamp) & ((event_rate.timestamps + initial_t) <= second_timestamp)
    indices = np.where(mask)[0]
    if indices.size == 0:
        continue

    # Find the index of the maximum event rate within this segment.
    local_peak_idx = indices[np.argmax(event_rate.samples[indices])]
    
    # Determine the time of the peak event rate (in seconds).
    peak_time = (event_rate.timestamps[local_peak_idx] + initial_t) / 1e6
    # Only annotate peaks that fall within the current xlim window.
    if peak_time < PLOT_RANGE[0] or peak_time > PLOT_RANGE[1]:
        continue

    # Interpolate to find the focal length at the time of the peak.
    focal_at_peak = np.interp(peak_time, events_seconds, focal_length_per_events)
    
    # Print the details in the terminal.
    print(f"Focal range [{first_timestamp/1e6:.2f} s, {second_timestamp/1e6:.2f} s] -> "
          f"Peak at {peak_time:.4f} s, Focal Length = {focal_at_peak:.4f}")
    
    # Draw a vertical dashed line at the event rate peak.
    ax1.axvline(x=peak_time, color='green', linestyle='--', lw=1.5)
    
    # Mark the intersection point on the focal length curve.
    ax1.plot(peak_time, focal_at_peak, 'o', color='green', markersize=8)
    
    # Annotate the intersection with the focal length value (4 decimals).
    y_top = ax1.get_ylim()[1] - 0.05 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
    ax1.text(peak_time, y_top, f"{focal_at_peak:.4f}", color='green',
             fontsize=10, ha="center", va="top", backgroundcolor="w", rotation=90)
    
    peak_times_array.append(peak_time)
    focal_points.append(focal_at_peak)

focal_timestamps_data = np.column_stack((peak_times_array, focal_points))
np.savetxt(f"{parent_directory}/focal_timestamps.txt", focal_timestamps_data, fmt="%.4f", header="timestamp focal_length", comments="")

# --- Final Plot Formatting ---
plt.title(f"Focal Length and Event Rate - Wavelength: {wavelength}nm")
plt.grid(True, linestyle='--', linewidth=0.5)
fig.tight_layout()
plt.savefig(f"{parent_directory}/focal_length_vs_eventrate.png")
plt.show()
