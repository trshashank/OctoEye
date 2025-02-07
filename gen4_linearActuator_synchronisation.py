import neuromorphic_drivers as nd
import serial
import time
import re

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
TEENSY_PORT = "/dev/ttyACM2"  # Adjust as needed
BAUD = 115200
LOG_FILENAME = "/media/samiarja/USB/Optical_characterisation/merged_log.csv"
WAIT_BEFORE_STARTING_ACTUATOR = 5.0  # seconds to wait after opening camera
CYCLE_COUNT = 10  # Number of cycles the actuator should run

# Regex to parse lines like: "T=908455 us, feedback=376"
TEENSY_LINE_REGEX = re.compile(r"^T=(\d+)\s+us,\s*feedback=(\d+)$")

# --------------------------------------------------------------
# Attempt to open Teensy (USB Serial)
# --------------------------------------------------------------
try:
    teensy = serial.Serial(TEENSY_PORT, BAUD, timeout=0.01)
    time.sleep(2)  # Wait for Teensy auto-reset
    print(f"Opened Teensy on {TEENSY_PORT}")
except Exception as e:
    print(f"Could not open Teensy port {TEENSY_PORT}: {e}")
    teensy = None

# --------------------------------------------------------------
# List available Prophesee devices & open log file
# --------------------------------------------------------------
nd.print_device_list()

log_file = open(LOG_FILENAME, "w", buffering=1)
log_file.write("entry_type,python_time_s,Gen4_time_us,teensy_time_us,feedback,rising\n")

# --------------------------------------------------------------
# MAIN LOOP: Open the camera, read packets, start actuator after 5 s
# --------------------------------------------------------------
with nd.open() as device:
    print("Gen4 opened. Starting event loop...")

    camera_start_time = time.time()
    actuator_started = False

    try:
        for status, packet in device:
            elapsed = time.time() - camera_start_time

            # 1) Start the actuator after the wait time
            if not actuator_started and elapsed >= WAIT_BEFORE_STARTING_ACTUATOR:
                print(f"Sending 'START {CYCLE_COUNT}' command to Teensy at t={elapsed:.2f}s")
                if teensy:
                    teensy.write(f"START {CYCLE_COUNT}\n".encode())  # Send cycle count
                actuator_started = True

            # 2) Read data from Teensy
            if teensy:
                while teensy.in_waiting > 0:
                    line = teensy.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        if line == "DONE":
                            print("Teensy finished all cycles. Shutting down...")
                            # device.close()  # Close the camera
                            log_file.close()  # Close the log file
                            teensy.close()  # Close Teensy serial
                            print("All systems safely shut down.")
                            exit(0)  # Exit program
                        match = TEENSY_LINE_REGEX.match(line)
                        if match:
                            teensy_time_us = int(match.group(1))
                            feedback = int(match.group(2))
                            now_s = time.time()
                            log_file.write(f"TEENSY,{now_s},NA,{teensy_time_us},{feedback},NA\n")
                            print(f"[Teensy] T={teensy_time_us} us, feedback={feedback}")
                        else:
                            print(f"[Teensy RAW] {line}")

            # 3) Process trigger events from the event camera
            if "trigger_events" in packet:
                for trig in packet["trigger_events"]:
                    Gen4_time_us = trig["t"]
                    rising_flag = trig["rising"]
                    now_s = time.time()
                    log_file.write(f"Gen4_TRIGGER,{now_s},{Gen4_time_us},NA,NA,{rising_flag}\n")
                    print(f"[Trigger Events] Gen4_time={Gen4_time_us}, rising={rising_flag}")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        log_file.close()
        print(f"Log saved to {LOG_FILENAME}")
