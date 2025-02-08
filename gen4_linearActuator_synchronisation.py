# import neuromorphic_drivers as nd
# import serial
# import time
# import re

# # --------------------------------------------------------------
# # CONFIG
# # --------------------------------------------------------------
# TEENSY_PORT = "/dev/ttyACM2"  # Adjust as needed
# BAUD = 115200
# LOG_FILENAME = "/media/samiarja/USB/Optical_characterisation/merged_log.csv"
# WAIT_BEFORE_STARTING_ACTUATOR = 5.0  # seconds to wait after opening camera
# CYCLE_COUNT = 10  # Number of cycles the actuator should run

# # Regex to parse lines like: "T=908455 us, feedback=376, unix_time=1700000000"
# TEENSY_LINE_REGEX = re.compile(r"^T=(\d+)\s+us,\s*feedback=(\d+),\s*unix_time=(\d+)$")

# # --------------------------------------------------------------
# # Attempt to open Teensy (USB Serial)
# # --------------------------------------------------------------
# try:
#     teensy = serial.Serial(TEENSY_PORT, BAUD, timeout=0.01)
#     time.sleep(2)  # Wait for Teensy auto-reset
#     print(f"Opened Teensy on {TEENSY_PORT}")
# except Exception as e:
#     print(f"Could not open Teensy port {TEENSY_PORT}: {e}")
#     teensy = None

# # --------------------------------------------------------------
# # List available Prophesee devices & open log file
# # --------------------------------------------------------------
# nd.print_device_list()

# log_file = open(LOG_FILENAME, "w", buffering=1)
# log_file.write("entry_type,python_time_s,Gen4_time_us,teensy_time_us,feedback,unix_time,rising\n")

# # --------------------------------------------------------------
# # MAIN LOOP: Open the camera, read packets, start actuator after 5 s
# # --------------------------------------------------------------
# with nd.open() as device:
#     print("Gen4 opened. Starting event loop...")
    
#     camera_start_time = time.time()
#     actuator_started = False
    
#     try:
#         for status, packet in device:
#             elapsed = time.time() - camera_start_time

#             # 1) Start the actuator after the wait time
#             if not actuator_started and elapsed >= WAIT_BEFORE_STARTING_ACTUATOR:
#                 print(f"Sending 'START {CYCLE_COUNT}' command to Teensy at t={elapsed:.2f}s")
#                 if teensy:
#                     teensy.write(f"START {CYCLE_COUNT}\n".encode())  # Send cycle count
#                 actuator_started = True

#             # 2) Read data from Teensy
#             if teensy:
#                 while teensy.in_waiting > 0:
#                     line = teensy.readline().decode("utf-8", errors="replace").strip()
#                     if line:
#                         if line == "DONE":
#                             print("Teensy finished all cycles. Shutting down...")
#                             # device.close()  # Close the camera
#                             log_file.close()  # Close the log file
#                             teensy.close()  # Close Teensy serial
#                             print("All systems safely shut down.")
#                             exit(0)  # Exit program
#                         match = TEENSY_LINE_REGEX.match(line)
#                         if match:
#                             teensy_time_us = int(match.group(1))
#                             feedback = int(match.group(2))
#                             unix_time = int(match.group(3))
#                             now_s = time.time()
#                             log_file.write(f"TEENSY,{now_s},NA,{teensy_time_us},{feedback},{unix_time},NA\n")
#                             print(f"[Teensy] T={teensy_time_us} us, feedback={feedback}, unix_time={unix_time}")
#                         else:
#                             print(f"[Teensy RAW] {line}")

#             # 3) Process trigger events from the event camera
#             if "trigger_events" in packet:
#                 for trig in packet["trigger_events"]:
#                     Gen4_time_us = trig["t"]
#                     rising_flag = trig["rising"]
#                     now_s = time.time()
#                     log_file.write(f"Gen4_TRIGGER,{now_s},{Gen4_time_us},NA,NA,NA,{rising_flag}\n")
#                     print(f"[Trigger Events] Gen4_time={Gen4_time_us}, rising={rising_flag}")

#     except KeyboardInterrupt:
#         print("Interrupted by user.")
#     finally:
#         log_file.close()
#         print(f"Log saved to {LOG_FILENAME}")


import serial
import time

# Configuration
TEENSY_PORT = "/dev/ttyACM0"  # Change as needed for your system.
BAUD = 115200
LOG_FILENAME = "teensy_data_log.txt"

# Open the serial port.
teensy = serial.Serial(TEENSY_PORT, BAUD, timeout=1)
time.sleep(0.5)
print(f"Opened Teensy on {TEENSY_PORT}")

# Send the Unix timestamp.
# (Note: The Teensy code adds micros() (in microseconds) to this value.)
current_unix_time = int(time.perf_counter_ns())

teensy.write(f"{current_unix_time}\n".encode())
print(f"Sent Unix timestamp: {current_unix_time}")

# Send the START command to begin the cycle.
teensy.write("START\n".encode())

# print("Waiting for data from Teensy...")

# # Open a file to log the data.
# with open(LOG_FILENAME, "w") as log_file:
#     while True:
#         line = teensy.readline().decode().strip()
#         # Uncomment the next line to see all raw incoming lines for debugging.
#         # print(f"DEBUG: Received raw line: '{line}'")
#         if line == "DATA_START":
#             continue
#         elif line == "DATA_END":
#             print("Data reception completed.")
#             break
#         elif line:
#             log_file.write(line + "\n")
#             print(f"Received: {line}")

# After data is received, send a RESET command.
teensy.write("RESET\n".encode())
teensy.close()
