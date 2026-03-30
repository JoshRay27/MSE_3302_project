//
//  Robot Right Hand - Finger Servo Control
//
//  Controls five servo motors (one per finger) on the robot's right hand.
//  Provides individual finger open/close functions and rock/paper/scissors gestures.
//
//  Modes (toggled via pushbutton):
//    Case 0: Stopped (red)
//    Case 1: Gesture loop - rock, paper, scissors on repeat (green)
//    Case 2: Rock paper scissors game (blue) [not yet implemented]
//
//  Naming convention uses "R" prefix throughout; left hand should mirror with "L".
//
//  Language: Arduino (C++)
//  Target:   ESP32
//

#include <Arduino.h>
#include <Adafruit_NeoPixel.h>

#ifndef R_P_C_H
#define R_P_C_H

void rRock();
void rScissors();
void rPaper();

#endif

// Function declarations
void doHeartbeat();
void ARDUINO_ISR_ATTR switchISR(void* arg);

// Switch structure (matches original)
struct Button {
  const int pin;
  uint32_t numberPresses;
  uint32_t nextPressTime;
  bool pressed;
};

// ─── Pin Assignments ──────────────────────────────────────────────────────────
const int cRThumbServo  = 16;
const int cRIndexServo  = 4;
const int cRMiddleServo = 15;
const int cRRingServo   = 21;
const int cRPinkyServo  = 22;
const int cSmartLED     = 23;   // Smart LED on GPIO23

// ─── Servo Timing Constants ───────────────────────────────────────────────────
const int cRThumbOpen   = 1000;
const int cRThumbClose  = 2100;
const int cRIndexOpen   = 1000;
const int cRIndexClose  = 2100;
const int cRMiddleOpen  = 1000;
const int cRMiddleClose = 2100;
const int cRRingOpen    = 1000;
const int cRRingClose   = 2100;
const int cRPinkyOpen   = 1000;
const int cRPinkyClose  = 2100;

// ─── Heartbeat Constants ──────────────────────────────────────────────────────
const int cHeartbeatInterval = 75;   // heartbeat update interval, in milliseconds
const int cSmartLEDCount     = 1;    // number of Smart LEDs in use
const int cDebounceDelay     = 250;  // switch debounce delay in milliseconds

// ─── Timer Variables ──────────────────────────────────────────────────────────
const int cGestureHoldInterval = 2000;
uint32_t timerCountGesture     = 0;
boolean timeUpGesture          = false;
uint32_t lastTime              = 0;
uint32_t lastHeartbeat         = 0;
uint32_t curMillis             = 0;

// ─── State Variables ──────────────────────────────────────────────────────────
int handMode                   = 0;   // current robot mode (0=stopped, 1=loop, 2=RPS game)
int gestureMode                = 0;   // tracks which gesture is next in the loop
Button modeButton              = {0, 0, 0, false};  // pushbutton on GPIO0

// ─── Smart LED Setup ──────────────────────────────────────────────────────────
Adafruit_NeoPixel SmartLEDs(cSmartLEDCount, cSmartLED, NEO_RGB + NEO_KHZ800);

uint32_t modeIndicator[3];  // populated in setup() once SmartLEDs is initialized

// Heartbeat brightness table (matches original)
unsigned char LEDBrightnessIndex  = 0;
unsigned char LEDBrightnessLevels[] = {0, 0, 0, 5, 15, 30, 45, 60, 75, 90, 105, 120, 135,
                                       150, 135, 120, 105, 90, 75, 60, 45, 30, 15, 5, 0};


// Setup message vars
String side = "";
String pos = "";
String msg ="";

// ─── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Set up Smart LED
  SmartLEDs.begin();
  SmartLEDs.clear();
  SmartLEDs.setPixelColor(0, SmartLEDs.Color(0, 0, 0));
  SmartLEDs.setBrightness(0);
  SmartLEDs.show();

  // Populate mode colours now that SmartLEDs is initialized
  modeIndicator[0] = SmartLEDs.Color(255, 0,   0);    // red   - stopped
  modeIndicator[1] = SmartLEDs.Color(0,   255, 0);    // green - gesture loop
  modeIndicator[2] = SmartLEDs.Color(0,   0,   255);  // blue  - RPS game

  // Attach all right hand finger servos (50 Hz, 14-bit resolution)
  ledcAttach(cRThumbServo,  50, 14);
  ledcAttach(cRIndexServo,  50, 14);
  ledcAttach(cRMiddleServo, 50, 14);
  ledcAttach(cRRingServo,   50, 14);
  ledcAttach(cRPinkyServo,  50, 14);

  // Set up pushbutton with interrupt
  pinMode(modeButton.pin, INPUT_PULLUP);
  attachInterruptArg(modeButton.pin, switchISR, &modeButton, FALLING);

  // Start in open (paper) position
  rPaper();
  Serial.println("System Ready. Press the button to start.");
  Serial.println("ESP32 Ready - Waiting for messages...");
}


// ─── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
  uint32_t curTime = micros();
  if (curTime - lastTime > 1000) {                    // wait 1 ms
    lastTime = curTime;

    // Gesture hold timer
    timerCountGesture += 1;
    if (timerCountGesture > cGestureHoldInterval) {
      timerCountGesture = 0;
      timeUpGesture = true;
    }

    // Handle button press — advance mode, reset state
    if (modeButton.pressed) {
      handMode++;
      if (handMode >= 3)
        handMode = 0;
      gestureMode = 0;
      timerCountGesture = 0;
      timeUpGesture = false;
      modeButton.pressed = false;
    }

    switch (handMode) {

      // Stopped — hand opens to resting paper position
      case 0:
        msg = Serial.readStringUntil('\n');
        msg.trim();
        if (msg.length() > 0) {
          Serial.print("Received: ");
          Serial.println(msg);

         // Optional: parse "left:3"
        int sep = msg.indexOf(':');
        if (sep != -1) {
          side = msg.substring(0, sep);
          pos = msg.substring(sep + 1);

          Serial.print("Side = ");
          Serial.println(side);
          Serial.print("Prediction = ");
          Serial.println(pos);
      }
    }
        if(side == "left"){
          if(pos == "rock"){
           // lRock();
          }else if(pos == "paper"){
           // lPaper();
          }else if(pos == "scissor"){
           // lScissor();
          }
        }else if(side == "right"){
          if(pos == "rock"){
            rRock();
          }else if(pos == "paper"){
            rPaper();
          }else if(pos == "scissors"){
            rScissors();
          }
        }
        break;

      // Gesture loop — cycles rock → paper → scissors every cGestureHoldInterval ms
      case 1:
        if (timeUpGesture) {
          timeUpGesture = false;
          switch (gestureMode) {
            case 0: rRock();     gestureMode++; break;
            case 1: rPaper();    gestureMode++; break;
            case 2: rScissors(); gestureMode = 0; break;
          }
        }
        break;

      // Rock paper scissors game — not yet implemented
      case 2:
        break;

    }
  }

  doHeartbeat();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Individual Finger Functions — Right Hand
// ═══════════════════════════════════════════════════════════════════════════════

void rOpenThumb()   { ledcWrite(cRThumbServo,  cRThumbOpen);   }
void rCloseThumb()  { ledcWrite(cRThumbServo,  cRThumbClose);  }

void rOpenIndex()   { ledcWrite(cRIndexServo,  cRIndexOpen);   }
void rCloseIndex()  { ledcWrite(cRIndexServo,  cRIndexClose);  }

void rOpenMiddle()  { ledcWrite(cRMiddleServo, cRMiddleOpen);  }
void rCloseMiddle() { ledcWrite(cRMiddleServo, cRMiddleClose); }

void rOpenRing()    { ledcWrite(cRRingServo,   cRRingOpen);    }
void rCloseRing()   { ledcWrite(cRRingServo,   cRRingClose);   }

void rOpenPinky()   { ledcWrite(cRPinkyServo,  cRPinkyOpen);   }
void rClosePinky()  { ledcWrite(cRPinkyServo,  cRPinkyClose);  }


// ═══════════════════════════════════════════════════════════════════════════════
//  Gesture Functions — Right Hand
// ═══════════════════════════════════════════════════════════════════════════════

void rRock() {
  rCloseThumb();
  rCloseIndex();
  rCloseMiddle();
  rCloseRing();
  rClosePinky();
}

void rPaper() {
  rOpenThumb();
  rOpenIndex();
  rOpenMiddle();
  rOpenRing();
  rOpenPinky();
}

void rScissors() {
  rCloseThumb();
  rOpenIndex();
  rOpenMiddle();
  rCloseRing();
  rClosePinky();
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Heartbeat LED
// ═══════════════════════════════════════════════════════════════════════════════

void doHeartbeat() {
  curMillis = millis();
  if ((curMillis - lastHeartbeat) > cHeartbeatInterval) {
    lastHeartbeat = curMillis;
    LEDBrightnessIndex++;
    if (LEDBrightnessIndex >= sizeof(LEDBrightnessLevels))
      LEDBrightnessIndex = 0;
    SmartLEDs.setBrightness(LEDBrightnessLevels[LEDBrightnessIndex]);
    SmartLEDs.setPixelColor(0, modeIndicator[handMode]);
    SmartLEDs.show();
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  Button ISR
// ═══════════════════════════════════════════════════════════════════════════════

void ARDUINO_ISR_ATTR switchISR(void* arg) {
  Button* s = static_cast<Button*>(arg);
  uint32_t pressTime = millis();
  if (pressTime > s->nextPressTime) {
    s->numberPresses += 1;
    s->pressed = true;
    s->nextPressTime = pressTime + cDebounceDelay;
  }
}