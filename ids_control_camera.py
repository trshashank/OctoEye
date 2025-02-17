from ids_peak import ids_peak
from ids_peak import ids_peak_ipl_extension
import numpy as np
from PIL import Image
import serial
import time

def main():
    # Initialize the library
    ids_peak.Library.Initialize()

    # Create a DeviceManager object
    device_manager = ids_peak.DeviceManager.Instance()

    try:
        # Register a temporary device-found callback
        device_found_callback = device_manager.DeviceFoundCallback(
            lambda found_device: print(
                "Found-Device-Callback: Key={}".format(
                    found_device.Key()), end="\n\n"))
        device_found_callback_handle = device_manager.RegisterDeviceFoundCallback(
            device_found_callback)

        # Update and then unregister the callback
        device_manager.Update()
        device_manager.UnregisterDeviceFoundCallback(
            device_found_callback_handle)

        # Exit if no device is found
        if device_manager.Devices().empty():
            print("No device found. Exiting Program.")
            return -1

        # Open the first device
        device = device_manager.Devices()[0].OpenDevice(ids_peak.DeviceAccessType_Control)
        remote_nodemap = device.RemoteDevice().NodeMaps()[0]

        # Load default camera settings
        remote_nodemap.FindNode("UserSetSelector").SetCurrentEntry("Default")
        remote_nodemap.FindNode("UserSetLoad").Execute()
        remote_nodemap.FindNode("UserSetLoad").WaitUntilDone()
        remote_nodemap.FindNode("ExposureTime").SetValue(2e+6)  # in microseconds

        # Set region-of-interest parameters
        x, y, width, height = 400, 700, 500, 500

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

        # Open the first data stream
        data_stream = device.DataStreams()[0].OpenDataStream()
        payload_size = remote_nodemap.FindNode("PayloadSize").Value()
        buffer_count_max = data_stream.NumBuffersAnnouncedMinRequired()

        # Allocate and announce buffers
        for buffer_count in range(buffer_count_max):
            buffer = data_stream.AllocAndAnnounceBuffer(payload_size)
            data_stream.QueueBuffer(buffer)

        # Lock writeable nodes during acquisition
        remote_nodemap.FindNode("TLParamsLocked").SetValue(1)

        print("Starting acquisition...")
        data_stream.StartAcquisition()
        remote_nodemap.FindNode("AcquisitionStart").Execute()
        remote_nodemap.FindNode("AcquisitionStart").WaitUntilDone()

        # Open the Arduino serial connection.
        # Adjust 'arduino_port' as needed (e.g., "COM3" for Windows or "/dev/ttyACM0" for Linux/Mac)
        arduino_port = "COM7"
        arduino_baudrate = 115200
        try:
            arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=2)
            time.sleep(2)  # Allow time for the Arduino to reset
            print("Connected to Arduino on", arduino_port)
        except Exception as e:
            print("Failed to connect to Arduino:", e)
            return -3

        print("Acquiring 100 images with motor movement...")
        for idx in range(24):
            try:
                # Wait for a finished/filled buffer event from the camera
                buffer = data_stream.WaitForFinishedBuffer(2000000)
                img = ids_peak_ipl_extension.BufferToImage(buffer)
                print(f"Image {idx} captured")

                # Wait 5 seconds before moving the motor
                time.sleep(5)

                # Command the Arduino to move the motor by sending "MOVE"
                arduino.write(b"MOVE\n")
                # Read the motor feedback value from Arduino (as a text line)
                feedback_line = arduino.readline().decode().strip()
                print(f"Received motor feedback: {feedback_line}")

                # Use the feedback value as the image filename
                filename = f"{feedback_line}.tiff"
                print(filename)

                # Clone the image and convert it to a PIL image before saving.
                # (Assumes the clone provides methods: Clone(), Width(), Height(), PixelFormat(), and get_data().)
                img_clone = img.Clone()
                # width_img = img_clone.Width()
                # height_img = img_clone.Height()
                # pixel_format = img_clone.PixelFormat()
                # img_buffer = img_clone.get_data()

                # if pixel_format == "Mono8":
                #     np_image = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height_img, width_img)
                #     pil_image = Image.fromarray(np_image, mode="L")
                # elif pixel_format in ["BGR8", "RGB8"]:
                #     np_image = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height_img, width_img, 3)
                #     if pixel_format == "BGR8":
                #         np_image = np_image[..., ::-1]  # Convert BGR to RGB
                #     pil_image = Image.fromarray(np_image, mode="RGB")
                # else:
                #     raise ValueError("Unhandled pixel format: " + pixel_format)

                # pil_image.save(filename, format="TIFF")
                # print(f"Saved image as {filename}")

                # Return the buffer to the data stream pool.
                data_stream.QueueBuffer(buffer)
            except Exception as e:
                print(f"Exception during image processing: {e}")

        # After processing all images, send a RESET command to the Arduino
        print("Sending RESET command to Arduino...")
        arduino.write(b"RESET\n")
        reset_feedback = arduino.readline().decode().strip()
        print("Motor reset feedback:", reset_feedback)

        print("Stopping acquisition...")
        remote_nodemap.FindNode("AcquisitionStop").Execute()
        remote_nodemap.FindNode("AcquisitionStop").WaitUntilDone()
        data_stream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
        data_stream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
        for buffer in data_stream.AnnouncedBuffers():
            data_stream.RevokeBuffer(buffer)
        remote_nodemap.FindNode("TLParamsLocked").SetValue(0)

    except Exception as e:
        print("EXCEPTION: " + str(e))
        return -2
    finally:
        ids_peak.Library.Close()
        if 'arduino' in locals() and arduino.is_open:
            arduino.close()

if __name__ == '__main__':
    main()
