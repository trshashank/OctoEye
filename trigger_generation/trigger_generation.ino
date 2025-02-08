// const int TRIGGER_PIN = 2;     // The Teensy pin to drive the EVK4 Trigger In
// const unsigned long TOTAL_TIME = 5000; // 5 seconds in ms
// const unsigned long PERIOD = 5;        // ms between pulses
// const unsigned long HIGH_TIME = 1;     // ms pin stays HIGH

// void setup() {
//   Serial.begin(115200);
//   pinMode(TRIGGER_PIN, OUTPUT);
//   digitalWrite(TRIGGER_PIN, LOW);  // Start LOW

//   Serial.println("Starting 5-second trigger pulse sequence at 200 Hz...");
// }

// void loop() {
//   unsigned long startTime = millis();

//   // Send pulses for 5 seconds total
//   while (millis() - startTime < TOTAL_TIME) {
//     // 1) Drive the pin HIGH for HIGH_TIME ms
//     digitalWrite(TRIGGER_PIN, HIGH);
//     delay(HIGH_TIME);

//     // 2) Then drive it LOW for the remainder of the period
//     digitalWrite(TRIGGER_PIN, LOW);
//     delay(PERIOD - HIGH_TIME);
//   }

//   Serial.println("Done pulsing. No more triggers are sent.");

//   // Stop pulsing forever
//   while (true) {
//     // You could do other tasks here or just idle
//   }
// }


// #include <Arduino.h>

// const int TRIGGER_PIN = 2;  // EVK4 trigger pin
// bool motionStarted = false;
// bool moveForward = true;
// unsigned long lastMoveTime = 0;
// const unsigned long MOVE_INTERVAL_MS = 2000;
// int cycleCount = 0;
// int maxCycles = 10;  // Default value if not received from Python

// void jrkSetTarget(uint16_t target) {
//   if (target > 4095) target = 4095;
//   Serial1.write(0xC0 + (target & 0x1F));
//   Serial1.write((target >> 5) & 0x7F);
// }

// uint16_t jrkGetFeedback() {
//   Serial1.write(0xE5);
//   Serial1.write(0x04);
//   Serial1.write(0x02);
//   while (Serial1.available() < 2);
//   return (uint16_t)(Serial1.read() + 256U * Serial1.read());
// }

// void setup() {
//   Serial.begin(115200);
//   Serial1.begin(9600);
//   pinMode(TRIGGER_PIN, OUTPUT);
//   digitalWrite(TRIGGER_PIN, LOW);
//   Serial.println("Teensy ready. Waiting for 'START X'...");
// }

// void loop() {
//   // 1) Read cycle count from serial command
//   if (Serial.available() > 0) {
//     String cmd = Serial.readStringUntil('\n');
//     cmd.trim();
//     if (cmd.startsWith("START")) {
//       int receivedCycles = cmd.substring(6).toInt();  // Extract cycle count
//       if (receivedCycles > 0) {
//         maxCycles = receivedCycles;
//       }
//       motionStarted = true;
//       lastMoveTime = millis();
//       Serial.print("Starting actuator cycles: ");
//       Serial.println(maxCycles);
//     }
//   }

//   if (!motionStarted) return;

//   // 2) Run the actuator for a limited number of cycles
//   if (millis() - lastMoveTime >= MOVE_INTERVAL_MS && cycleCount < maxCycles) {
//     lastMoveTime = millis();
//     uint16_t target = moveForward ? 800 : 100;
//     jrkSetTarget(target);
//     digitalWrite(TRIGGER_PIN, HIGH);
//     delayMicroseconds(10);
//     digitalWrite(TRIGGER_PIN, LOW);
//     uint16_t fb = jrkGetFeedback();
//     unsigned long long sys_time = micros(); // Get high-precision timestamp in microseconds

//     Serial.print("T=");
//     Serial.print(micros());
//     Serial.print(" us, feedback=");
//     Serial.print(fb);
//     Serial.print(", sys_time=");
//     Serial.println(sys_time);

//     moveForward = !moveForward;
//     cycleCount++;
//   }

//   // 3) After final cycle, stop everything
//   if (cycleCount >= maxCycles) {
//     Serial.println("DONE");
//     motionStarted = false;

//     // Set actuator to stop position (adjust as needed)
//     jrkSetTarget(2048);  // Neutral position or stop signal

//     // Ensure trigger pin is LOW
//     digitalWrite(TRIGGER_PIN, LOW);

//     // Wait to ensure Python receives the "DONE" message before stopping the loop
//     delay(1000);

//     // Stop execution
//     while (true) { delay(1000); }  // Prevents further execution
//   }
// }

// Arduino (Teensy) Code – unchanged
#include <Arduino.h>
#include <TimeLib.h>

const int TRIGGER_PIN = 2;  // EVK4 trigger pin
const unsigned long MOVE_INTERVAL_MS = 2000;
const int maxCycles = 10;  // Number of cycles to run before stopping

bool moveForward = true;
unsigned long lastMoveTime = 0;
int cycleCount = 0;
bool startCommandReceived = false;
unsigned long long unix_time_offset = 0;  // Will store the reference Unix time

void jrkSetTarget(uint16_t target) {
    if (target > 3500) target = 3500;
    Serial1.write(0xC0 + (target & 0x1F));
    Serial1.write((target >> 5) & 0x7F);
}

uint16_t jrkGetFeedback() {
    Serial1.write(0xE5);
    Serial1.write(0x04);
    Serial1.write(0x02);
    while (Serial1.available() < 2);
    return (uint16_t)(Serial1.read() + 256U * Serial1.read());
}

void setup() {
    Serial.begin(115200);
    Serial1.begin(9600);
    pinMode(TRIGGER_PIN, OUTPUT);
    digitalWrite(TRIGGER_PIN, LOW);

    Serial.println("Teensy ready. Waiting for Unix time from Python...");

    while (Serial.available() == 0) {
        // Wait for Python to send a reference Unix timestamp
    }

    // Read the Unix timestamp as a string
    String input = Serial.readStringUntil('\n');
    unix_time_offset = strtoull(input.c_str(), NULL, 10);  // Converts string to 64-bit integer

    Serial.print("Received Unix timestamp: ");
    Serial.println(unix_time_offset);
}

// Function to return synchronized Unix timestamp
unsigned long long getUnixTimestamp() {
    return unix_time_offset + micros();  // Both values are in microseconds
}

void loop() {
    // 1️⃣ Wait for Python to send "START"
    if (!startCommandReceived && Serial.available() > 0) {
        String cmd = Serial.readStringUntil('\n');
        cmd.trim();
        if (cmd.equalsIgnoreCase("START")) {
            startCommandReceived = true;
            Serial.println("READY");  // Only print once
        }
    }

    if (!startCommandReceived) {
        return;  // Do nothing until START is received
    }

    // 2️⃣ Run the actuator movement
    if (millis() - lastMoveTime >= MOVE_INTERVAL_MS && cycleCount < maxCycles) {
        lastMoveTime = millis();
        uint16_t target = moveForward ? 800 : 100;
        jrkSetTarget(target);
        digitalWrite(TRIGGER_PIN, HIGH);
        delayMicroseconds(10);
        digitalWrite(TRIGGER_PIN, LOW);

        // Get Unix timestamp and feedback
        unsigned long long unix_time = getUnixTimestamp();
        uint16_t feedback = jrkGetFeedback(); 

        // Print values in CSV format for Python
        Serial.print(unix_time);
        Serial.print(", ");
        Serial.println(feedback);

        moveForward = !moveForward;
        cycleCount++;
    }

      // 3️⃣ After finishing cycles, reset Teensy to wait for a new START
    if (cycleCount >= maxCycles) {
        Serial.println("DONE");  // Notify Python that the cycle is complete
        Serial.flush();          // Wait until all data is transmitted

        jrkSetTarget(800);  // Stop the motor
        digitalWrite(TRIGGER_PIN, LOW);
        delay(2000);        // Allow time for any final communications

        Serial.println("Restarting Teensy...");
        Serial.flush();      // Ensure this message is sent as well
        delay(200);          // A short delay before reset
        
        SCB_AIRCR = 0x05FA0004;  // Soft reset Teensy
    }

}






