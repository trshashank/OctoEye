import serial
import time

'''
/*
 * CODE TO AUTOMATE EVENT CAMERA
 */

#include <Arduino.h>

// Remove EEPROM since we are using an array

const int TRIGGER_PIN = 2;            // EVK4 trigger pin
const unsigned long MOVE_INTERVAL_MS = 15000; // 15 seconds
const int maxCycles = 60;             // Number of cycles to run before stopping

bool moveForward = true;
unsigned long lastMoveTime = 0;
int cycleCount = 0;
bool startCommandReceived = false;
unsigned long long unix_time_offset = 0;  // Reference Unix time received from Python

struct DataPoint {
  unsigned long long timestamp;       // In microseconds
  uint16_t feedback;
};

DataPoint dataPoints[maxCycles];      // Array to store the data points

// Send collected data over Serial
void sendDataToPython() {
  Serial.println("DATA_START");
  for (int i = 0; i < maxCycles; i++) {
    Serial.print(dataPoints[i].timestamp);
    Serial.print(",");
    Serial.println(dataPoints[i].feedback);
  }
  Serial.println("DATA_END");
  Serial.flush();
}

// These functions communicate with the motor controller on Serial1.
void jrkSetTarget(uint16_t target) {
  if (target > 3400) target = 3400;
  Serial1.write(0xC0 + (target & 0x1F));
  Serial1.write((target >> 5) & 0x7F);
}

uint16_t jrkGetFeedback() {
  Serial1.write(0xE5);
  Serial1.write(0x04);
  Serial1.write(0x02);
  // Wait until two bytes are available:
  while (Serial1.available() < 2);
  // First byte is the high part:
  uint8_t highByte = Serial1.read();
  uint8_t lowByte  = Serial1.read();
  return (uint16_t)(highByte + 256U * lowByte);
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  pinMode(TRIGGER_PIN, OUTPUT);
  // digitalWrite(TRIGGER_PIN, LOW);

  Serial.println("Teensy ready. Waiting for Unix time from Python...");

  // Wait for a Unix timestamp from Python (make sure the Python code sends a newline-terminated timestamp)
  while (!Serial.available());
  String input = Serial.readStringUntil('\n');
  unix_time_offset = strtoull(input.c_str(), NULL, 10);

  Serial.print("Received Unix timestamp: ");
  Serial.println(unix_time_offset);
}

unsigned long long getUnixTimestamp() {
  // Convert micros() (microseconds) to nanoseconds before adding the offset.
  return unix_time_offset + ((unsigned long long)micros() * 1000ULL);
}


void loop() {
  // Wait for the "START" command if not already received
  if (!startCommandReceived && Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.equalsIgnoreCase("START")) {
      startCommandReceived = true;
      Serial.println("READY");
    }
  }
  
  if (!startCommandReceived) {
    return;
  }

  // Every MOVE_INTERVAL_MS milliseconds, execute a cycle (up to maxCycles)
  if (millis() - lastMoveTime >= MOVE_INTERVAL_MS && cycleCount < maxCycles) {
    lastMoveTime = millis();
    uint16_t target = moveForward ? 3200 : 2000;
    jrkSetTarget(target);
    
    // Pulse the trigger pin
    digitalWrite(TRIGGER_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGGER_PIN, LOW);

    // Record the current Unix time and the feedback from the controller
    unsigned long long unix_time = getUnixTimestamp();
    uint16_t feedback = jrkGetFeedback();

    // Save the data in our array
    dataPoints[cycleCount].timestamp = unix_time;
    dataPoints[cycleCount].feedback = feedback;

    moveForward = !moveForward;
    cycleCount++;
  }

  
  // Once all cycles are complete, send the data to Python.
  if (cycleCount >= maxCycles) {
    sendDataToPython();
    Serial.println("DONE");
    Serial.flush();
  }
}
'''

# Configuration
TEENSY_PORT = "COM7" #"/dev/ttyACM0"  # Change as needed for your system.
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
