import json
import numpy as np
import octoeye
import matplotlib.pyplot as plt

# ------------------------------
# CONFIGURATION / FILE PATHS
# ------------------------------

event_filename = "2025-02-08T11-14-11Z"
parent_directory = "/home/samiarja/Desktop/PhD/Code/gitlib/gen4/recordings/red_dot"
gen4_triggers_path = f"{parent_directory}/00051508_control_events.jsonl"

teensy_log_path = f"{parent_directory}/teensy_data_log.txt"
output_with_focal_length_path = f"{parent_directory}/teensy_data_log_aligned_timestamp.txt"

width, height, events = octoeye.read_es_file(f"{parent_directory}/{event_filename}.es")
sensor_size = (width, height)

# ------------------------------
# PARAMETERS FOR DYNAMIC CONVERSION
# ------------------------------
feedback_min = 284     # At this raw feedback, camera is farthest away.
feedback_max = 3965    # At this raw feedback, camera is closest.
distance_at_min_mm = 92  # When feedback is at minimum, camera is 920 mm (92 cm) away from the ball lens.
# The dynamic conversion formula is:
#   focal_length_mm = (feedback_max - feedback_value) / (feedback_max - feedback_min) * distance_at_min_mm
feedback_range = feedback_max - feedback_min

# ------------------------------
# LOAD TEENSY DATA
# ------------------------------
# Assuming the Teensy log file has two columns: raw_timestamp (ns) and feedback (raw value)
teensy_data = np.loadtxt(teensy_log_path, delimiter=",", dtype=int)
raw_timestamps = teensy_data[:, 0]   # in nanoseconds
feedback_values = teensy_data[:, 1]

# ------------------------------
# PARSE GEN4 TRIGGERS (EVENT CAMERA)
# ------------------------------
rising_events = []
falling_events = []
start_recording_info = {}

# Read and parse the JSONL file
with open(gen4_triggers_path, "r") as file:
    for line in file:
        data = json.loads(line.strip())
        
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
offset = raw_timestamps[0] - rising_events[0]['system_timestamp']
print(f"Computed offset (Teensy - Camera) = {offset} ns")
print("............................................")

# ------------------------------
# COMPUTE CORRECTED TIMESTAMPS
# ------------------------------
# Apply offset correction and convert from nanoseconds to seconds.
corrected_timestamps = raw_timestamps - offset
corrected_timestamps_sec = corrected_timestamps / 1_000_000_000.0

# ------------------------------
# CONVERT FEEDBACK TO FOCAL LENGTH (in mm)
# ------------------------------
# Dynamic conversion: as the raw feedback increases from 284 to 3965, the camera moves from 920 mm away to 0 mm.
focal_length_mm = (feedback_max - feedback_values) / feedback_range * distance_at_min_mm

# Save new log file with four columns: raw_timestamp, feedback, corrected_timestamp_sec, focal_length_mm
data_with_focal_length = np.column_stack((raw_timestamps, feedback_values, corrected_timestamps_sec * 1_000_000_000.0, focal_length_mm))
header_line_focal = "raw_timestamp,feedback,corrected_timestamp_sec,focal_length_mm"
np.savetxt(output_with_focal_length_path, data_with_focal_length, delimiter=",", fmt="%d,%d,%.9f,%.3f",
           header=header_line_focal, comments="")
print(f"Corrected data log with focal length saved to {output_with_focal_length_path}")

# ------------------------------
# PLOT FOCAL LENGTH VS. CORRECTED TIMESTAMP
# ------------------------------
# plt.style.use('seaborn-darkgrid')

##### label the events with the focal length using the correct timstamp
# find the closest real event timestamp based on the corrected teensy unix timestamp
# label all the either from 35.815 to 13.421 or from 13.421 to 35.815

plt.figure(figsize=(12, 8))

# Plot the dynamic focal length data
# (Choose a visually appealing color, here a deep blue from the default palette)
plt.plot(corrected_timestamps_sec, focal_length_mm, 'o-', 
         color='#1f77b4', markersize=8, linewidth=2, label="Focal Length (mm)")

# Add horizontal dashed lines for the maximum and minimum focal lengths
# Maximum: 92 cm (camera farthest away)
plt.axhline(y=max(focal_length_mm), color='black', linestyle='--', linewidth=1.5, 
            label=f'Max Focal Length ({max(focal_length_mm):.2f} mm)')
# Minimum: 0 cm (camera closest)
plt.axhline(y=min(focal_length_mm), color='black', linestyle='--', linewidth=1.5, 
            label=f'Min Focal Length ({min(focal_length_mm):.2f} mm)')


# Labeling and titling
plt.xlabel("Timestamp (Aligned) - seconds", fontsize=14, fontweight='bold')
plt.ylabel("Focal Length - mm", fontsize=14, fontweight='bold')
# plt.title("Dynamic Focal Length vs. Time", fontsize=16, fontweight='bold')

# Improve the legend appearance
plt.legend(fontsize=12, frameon=True, edgecolor='black')

# Optional: Fine-tune grid lines and layout for clarity
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()

plt.savefig(f"{parent_directory}/focal_length_vs_time.png")
