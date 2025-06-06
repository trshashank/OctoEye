import os
from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from PIL import Image
import serial
import time
import matplotlib.pyplot as plt
import tifffile

'''
/*
 * Updated Arduino code to move the motor from 2400 to 3600 in steps of 50.
 * It starts by forcing the motor to the minimum position (2400) before any frames are captured.
 * Then, it waits for commands ("MOVE" to increment the motor and "RESET" to return to 2400).
 * This code is used to automate a greyscale sensor.
 */

#include <Arduino.h>

const int TRIGGER_PIN = 2;            // EVK4 trigger pin

// Motor range and step parameters
const uint16_t MIN_POSITION = 2800;
const uint16_t MAX_POSITION = 3400;
const uint16_t STEP = 1;
uint16_t currentPosition = MIN_POSITION;  // Start at MIN_POSITION

// Function to send target value to the motor controller over Serial1.
void jrkSetTarget(uint16_t target) {
  if (target > MAX_POSITION) {
    target = MAX_POSITION;
  }
  Serial1.write(0xC0 + (target & 0x1F));
  Serial1.write((target >> 5) & 0x7F);
}

// Function to read back the motor’s feedback.
uint16_t jrkGetFeedback() {
  Serial1.write(0xE5);
  Serial1.write(0x04);
  Serial1.write(0x02);
  // Wait until two bytes are available:
  while (Serial1.available() < 2);
  uint8_t highByte = Serial1.read();
  uint8_t lowByte  = Serial1.read();
  return (uint16_t)(highByte + 256U * lowByte);
}

void setup() {
  Serial.begin(115200);   // Communication with Python
  Serial1.begin(9600);    // Communication with motor controller
  //pinMode(TRIGGER_PIN, OUTPUT);
  //digitalWrite(TRIGGER_PIN, LOW);
  
  // Force motor to start at MIN_POSITION before capturing the first frame.
  currentPosition = MIN_POSITION;
  jrkSetTarget(currentPosition);
  delay(100);  // Allow motor to start moving
  uint16_t initFeedback = jrkGetFeedback();
  Serial.print("Motor initialized to ");
  Serial.println(initFeedback);
  
  Serial.println("Arduino ready. Awaiting commands (MOVE/RESET).");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.equalsIgnoreCase("MOVE")) {
      // Increment current position by STEP if we haven't reached MAX_POSITION
      if (currentPosition < MAX_POSITION) {
        currentPosition += STEP;
      } else {
        Serial.println("MAX_REACHED");
        return;
      }
      
      // Set the motor to the new target
      jrkSetTarget(currentPosition);
      
      // Short delay to allow the motor to start moving
      delay(100);
      
      // Get the feedback value from the motor controller
      uint16_t feedback = jrkGetFeedback();
      
      // Send the feedback value back to Python (as a line of text)
      Serial.println(feedback);
    }
    else if (cmd.equalsIgnoreCase("RESET")) {
      // Reset motor to MIN_POSITION (2400)
      currentPosition = MIN_POSITION;
      jrkSetTarget(currentPosition);
      delay(100);
      uint16_t feedback = jrkGetFeedback();
      Serial.println(feedback);
    }
  }
}
'''


def main():
    # Ensure the frames folder exists.
    wavelength = "950"
    main_path = "D:/Optical_characterisation/hyperspectral_high_resolution"
    if not os.path.exists(f"{main_path}/{wavelength}"):
        os.makedirs(f"{main_path}/{wavelength}")
    
    # Initialize the library.
    ids_peak.Library.Initialize()
    device_manager = ids_peak.DeviceManager.Instance()
    
    try:
        # Register and then unregister a temporary device-found callback.
        device_found_callback = device_manager.DeviceFoundCallback(
            lambda found_device: print("Found-Device-Callback: Key={}".format(found_device.Key()), end="\n\n"))
        device_found_callback_handle = device_manager.RegisterDeviceFoundCallback(device_found_callback)
        device_manager.Update()
        device_manager.UnregisterDeviceFoundCallback(device_found_callback_handle)
    
        if device_manager.Devices().empty():
            print("No device found. Exiting Program.")
            return -1
    
        # Open the first device.
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        remote_nodemap = device.RemoteDevice().NodeMaps()[0]
    
        # Load default camera settings.
        remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        remote_nodemap.FindNode("UserSetLoad").Execute()
        remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
        remote_nodemap.FindNode("PixelFormat").SetCurrentEntry("Mono12")

        remote_nodemap.FindNode("ExposureTime").SetValue(2e+6)  # in microseconds
        
        # Set the camera to continuous acquisition mode (if available).
        try:
            remote_nodemap.FindNode("AcquisitionMode").SetCurrentEntry("Continuous")
            print("Acquisition mode set to Continuous")
        except Exception as e:
            print("Failed to set AcquisitionMode to Continuous:", e)
    
        # Set region-of-interest parameters.
        x, y, width, height = 700, 400, 500, 500
        x_min = remote_nodemap.FindNode("OffsetX").Minimum()
        y_min = remote_nodemap.FindNode("OffsetY").Minimum()
        w_min = remote_nodemap.FindNode("Width").Minimum()
        h_min = remote_nodemap.FindNode("Height").Minimum()
        remote_nodemap.FindNode("OffsetX").SetValue(x_min)
        remote_nodemap.FindNode("OffsetY").SetValue(y_min)
        remote_nodemap.FindNode("Width").SetValue(w_min)
        remote_nodemap.FindNode("Height").SetValue(h_min)
        x_max = remote_nodemap.FindNode("OffsetX").Maximum()
        y_max = remote_nodemap.FindNode("OffsetY").Maximum()
        w_max = remote_nodemap.FindNode("Width").Maximum()
        h_max = remote_nodemap.FindNode("Height").Maximum()
    
        if (x < x_min) or (y < y_min) or (x > x_max) or (y > y_max):
            print("Error: x and y values out of range")
            return False
        elif (width < w_min) or (height < h_min) or ((x + width) > w_max) or ((y + height) > h_max):
            print("Error: width and height values out of range")
            return False
        else:
            print("Setting final ROI")
            remote_nodemap.FindNode("OffsetX").SetValue(x)
            remote_nodemap.FindNode("OffsetY").SetValue(y)
            remote_nodemap.FindNode("Width").SetValue(width)
            remote_nodemap.FindNode("Height").SetValue(height)
    
        remote_nodemap.FindNode("Gain").SetValue(1.0)
        
    
        # Open the first data stream.
        data_stream = device.DataStreams()[0].OpenDataStream()
        payload_size = remote_nodemap.FindNode("PayloadSize").Value()
        buffer_count_max = data_stream.NumBuffersAnnouncedMinRequired()
    
        # Allocate and announce buffers.
        for _ in range(buffer_count_max):
            buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
            data_stream.QueueBuffer(buffer)
    
        # Lock writeable nodes during acquisition.
        remote_nodemap.FindNode("TLParamsLocked").SetValue(1)
    
        print("Starting acquisition...")
        data_stream.StartAcquisition()
        remote_nodemap.FindNode("AcquisitionStart").Execute()
        remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()
    
        # Open the Arduino serial connection.
        arduino_port = "COM7"  # Adjust as needed.
        arduino_baudrate = 115200
        try:
            arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=5)
            time.sleep(2)  # Allow time for Arduino to reset.
            print("Connected to Arduino on", arduino_port)
        except Exception as e:
            print("Failed to connect to Arduino:", e)
            return -3
    
        # Define the motor positions from 2600 to 3400 in steps of 50.
        steps = 1
        positions = list(range(2800, 3250 + 1, steps))
    
        for pos in positions:
            try:
                # time.sleep(5)
                # Capture frame.
                buffer = data_stream.WaitForFinishedBuffer(2000000)
                img = ids_peak_ipl_extension.BufferToImage(buffer)
                print(f"Image captured at motor feedback {pos}")

                # time.sleep(2)
    
                # Save frame as TIFF without compression.
                img_clone = img
                img_buffer = img_clone.get_numpy()
                # img_2d = np.squeeze(img_buffer, axis=2)  # Convert from (500,500,1) to (500,500)

                # img_2d[img_2d > 10000] = 0

                tiff_filename = f"{main_path}/{wavelength}/{pos}.tiff"
                tifffile.imwrite(tiff_filename, img_buffer, compression='none')
                print(f"Saved image as {tiff_filename}")

                time.sleep(1)
    
                # Return the buffer to the data stream pool.
                data_stream.QueueBuffer(buffer)
    
                # If this is not the last position, send the MOVE command to the Arduino.
                if pos != positions[-1]:
                    arduino.write(b"MOVE\n")
                    arduino.flush()
                    # Wait a moment for the motor to move.
                    time.sleep(5)
    
            except Exception as e:
                print(f"Exception during image processing: {e}")
    
        # Optionally, send a RESET command to return the motor to 2600.
        print("Sending RESET command to Arduino...")
        arduino.write(b"RESET\n")
        arduino.flush()
        time.sleep(1)
        print("Motor reset command sent.")
    
        print("Stopping acquisition...")
        remote_nodemap.FindNode("AcquisitionStop").Execute()
        remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
        data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in data_stream.AnnouncedBuffers():
            data_stream.RevokeBuffer(buffer)
        remote_nodemap.FindNode("TLParamsLocked").SetValue(0)
    
    except Exception as e:
        print("EXCEPTION:", str(e))
        return -2
    finally:
        ids_peak.Library.Close()
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()
    
if __name__ == '__main__':
  time_start = time.time()
  main()
  time_end = time.time()
  print(f"Time taken: {(time_end - time_start)/60} minutes")
