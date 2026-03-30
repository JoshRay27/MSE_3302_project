#include <Arduino.h>
#include <Adafruit_NeoPixel.h>


//Motor Control Pins
const int IN1Pin[] = {26, 16};                               // GPIO pin(s) for IN1 for left and right motors (A, B)
const int IN2Pin[] = {27, 17};                               // GPIO pin(s) for IN2 for left and right motors (A, B) 
const int PWMFreq = 20000;
const int PWMRes = 8; 
int currentPWM = 0; 
const int NumMotors = 2;                             // Number of DC motors

// Encoder Settings
struct Encoder {
  const int PinA;
  const int PinB;
  volatile long pos;
};
Encoder encoder [] = {{35, 32, 0},
                      {33, 25, 0}}; 

//SmartLED
const int cSmartLED = 23;      // Pin for the NeoPixel
const int cSmartLEDCount = 1;  // Number of LEDs
Adafruit_NeoPixel SmartLEDs(cSmartLEDCount, cSmartLED, NEO_RGB + NEO_KHZ800);

uint32_t modeIndicator[2] = {
  SmartLEDs.Color(255, 0, 0),   // Mode 0: RED - System Idle 
  SmartLEDs.Color(0, 255, 0)    // Mode 1: GREEN - System Active
};

// POT speed control
const int cPotPin = 36;                       // GPIO 36 connects to the white PIHER potentiometer
const int cMinPWM = 150;                      // Minimum PWM value needed to actually turn the motor
const int cMaxPWM = pow(2,PWMRes) - 1;       // Maximum 8-bit PWM speed
int driveSpeed = 0;                           // Variable to store the final calculated speed

int myNumber = 0;                             // Varible to store user input

// DC Motor distance traveled logic
const float dcEncoderPerRev = 1096.0;                                     // Encoder count per revolution
const float pitchDiameter = 2.25;                                         // pitch diameter in cm
float Circumference = 3.143 * pitchDiameter;
float distancePerCount = Circumference / dcEncoderPerRev;                 // Distance gear travels per encoder count
float rackDistance = 18.0 / distancePerCount;
volatile long removeDistance[NumMotors];                                  // stores encoder value for arm removal distance


// Button 
const long cDebounceDelay = 170;
struct Button {
  const int pin;
  volatile uint32_t numberPresses;
  uint32_t nextPressTime;
  volatile bool pressed;
};
Button modeButton = {0, 0, 0, false}; 


// State and Timing Varibles
uint32_t lastTime = 0; 

// State (Controlled by Button)
uint8_t robotModeIndex = 0; // 0 = Idle, 1 = Active

// Timer
uint32_t timerCount1sec = 0; 
bool timeUp1sec = false;

// Message Var
String pred = "";
String side = "";

// Motor Vars
long currentPos[NumMotors];

// Function declarations
void setMotor(int dir, int pwm, int in1, int in2);
void ARDUINO_ISR_ATTR encoderISR(void* arg);
void ARDUINO_ISR_ATTR buttonISR(void* arg);
void moveRightBack(int driveSpeed);
void moveLeftBack(int driveSpeed);
void returnRightBack(int driveSpeed);
void returnLeftBack(int driveSpeed);


void setup() {
  Serial.begin(115200);

  // setup motors with encoders
  for (int k = 0; k < NumMotors; k++) {
    ledcAttach(IN1Pin[k], PWMFreq, PWMRes);                         // setup INT1 GPIO PWM Channel
    ledcAttach(IN2Pin[k], PWMFreq, PWMRes);                         // setup INT2 GPIO PWM Channel
    pinMode(encoder[k].PinA, INPUT);                                   // configure GPIO for encoder Channel A input
    pinMode(encoder[k].PinB, INPUT);                                   // configure GPIO for encoder Channel B input
    attachInterruptArg(encoder[k].PinA, encoderISR, &encoder[k], RISING); // configure encoder to trigger interrupt with each rising edge on Channel A
  }

  // Setup SmartLED
  SmartLEDs.begin();
  SmartLEDs.setBrightness(50); // Set a visible brightness level (0-255)
  SmartLEDs.setPixelColor(0, modeIndicator[robotModeIndex]); // Initialize to Red
  SmartLEDs.show();

  // Setup Button
  pinMode(modeButton.pin, INPUT_PULLUP);
  attachInterruptArg(modeButton.pin, buttonISR, &modeButton, FALLING);

  Serial.println("System Ready. Press the button to start.");
  Serial.println("ESP32 Ready - Waiting for messages...");
}


void loop() {
  uint32_t curTime = micros();
  if (curTime - lastTime > 1000) { // 1ms Tick Timer
    lastTime = curTime;

  // Encoder Read
    noInterrupts();
    for (int k = 0; k < NumMotors; k++) {
      currentPos[k] = encoder[k].pos;       // read and store current motor position
    }
    interrupts();
    // Timing logic 
    timerCount1sec++; 
    if (timerCount1sec >= 1000) {  // 1 second timer
      timerCount1sec = 0; 
      timeUp1sec = true; 
      }
    
    // Button logic (Mode Switching)
    if (modeButton.pressed) {
      modeButton.pressed = false; // Acknowledge press
      
      robotModeIndex++;
      robotModeIndex = robotModeIndex % 2; // Keep mode between 0 and 1

      // Update the LED color immediately
      SmartLEDs.setPixelColor(0, modeIndicator[robotModeIndex]);
      SmartLEDs.show();

      Serial.printf("Mode switched to: %d\n", robotModeIndex);
    }
  
    
    // modes 
    // 0 = Default after power up/reset (Robot is stopped)
    // 1 = Press mode button once to enter rack movement mode
    switch (robotModeIndex){
    case 0: // Robot stopped
      setMotor(0, 0, IN1Pin[0], IN2Pin[0]);
      setMotor(0, 0, IN1Pin[1], IN2Pin[1]);
      encoder[0].pos = 0;
      encoder[1].pos = 0;
      myNumber = -1;
    break;

    
    case 1: // Motor actuating logic       
      // Read the physical knob (0 to 4095)
      int potValue = analogRead(cPotPin);
      
      // Map the knob position to your safe motor speed range
      driveSpeed = map(potValue, 0, 4095, cMinPWM, cMaxPWM);
        if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');  // read one full line

    msg.trim();  // remove whitespace

    if (msg.length() > 0) {
      Serial.print("Received: ");
      Serial.println(msg);

      // Optional: parse "left:3"
      int sep = msg.indexOf(':');
      if (sep != -1) {
        side = msg.substring(0, sep);
        pred = msg.substring(sep + 1);

        Serial.print("Side = ");
        Serial.println(side);
        Serial.print("Prediction = ");
        Serial.println(pred);
      }
    }
  }
      if(side == "left"){
      // Move left rack back
        if(pred == "back"){
          moveLeftBack(driveSpeed);
        }

        // Return Left rack to intial position
        else if(pred == "forward" ){
          returnLeftBack(driveSpeed);
        }
      }
      if (side == "right"){
              // Move right rack back
        if(pred == "back"){
          moveRightBack(driveSpeed);
        }
        
        // Return right rack to inital position
        else if(pred == "forward"){
          returnRightBack(driveSpeed);
        }
      }

      // idle state
      if (pred == "idle"){
        idle();
      }

      // goes back to robotModeIndex = 0
      if(pred == 0 ){
        robotModeIndex = 0;
        SmartLEDs.setPixelColor(0, modeIndicator[robotModeIndex]);
        SmartLEDs.show();
      }

    break;

    }
  }
} 



void moveRightBack(int driveSpeed){
        setMotor(-1, driveSpeed, IN1Pin[0], IN2Pin[0]);
        if (timeUp1sec) {
          Serial.println("left motor moving CCW");
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[0], driveSpeed);
          timeUp1sec = false; 
          timerCount1sec = 0;
        }
        if (encoder[0].pos <= -1*rackDistance){
          setMotor(0, 0, IN1Pin[0], IN2Pin[0]);
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[0], driveSpeed);
          removeDistance[0] = encoder[0].pos;
          encoder[0].pos = 0;
          resetMsg();
        }
}

void returnRightBack(int driveSpeed){
        setMotor(1, driveSpeed, IN1Pin[0], IN2Pin[0]);
        if (timeUp1sec) {
          Serial.println("motor moving CW");
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[0], driveSpeed);
          timeUp1sec = false; 
          timerCount1sec = 0;
        }
        if(encoder[0].pos >= -1*removeDistance[0]){
          setMotor(0, 0, IN1Pin[0], IN2Pin[0]);
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[0], driveSpeed);
          encoder[0].pos = 0;
          resetMsg();
        }
}

void moveLeftBack(int driveSpeed){
        setMotor(-1, driveSpeed, IN1Pin[1], IN2Pin[1]);
        if (timeUp1sec) {
          Serial.println("left motor moving CCW");
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[1], driveSpeed);
          timeUp1sec = false; 
          timerCount1sec = 0;
        }
        if (encoder[1].pos >= rackDistance){
          setMotor(0, 0, IN1Pin[1], IN2Pin[1]);
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[1], driveSpeed);
          removeDistance[1] = encoder[1].pos;
          encoder[1].pos = 0;
          resetMsg();
        }
}

void returnLeftBack(int driveSpeed){
        setMotor(1, driveSpeed, IN1Pin[1], IN2Pin[1]);
        if (timeUp1sec) {
          Serial.println("motor moving CW");
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[1], driveSpeed);
          timeUp1sec = false; 
          timerCount1sec = 0;
        }
        if(encoder[1].pos <= -1*removeDistance[1]){
          setMotor(0, 0, IN1Pin[1], IN2Pin[1]);
          Serial.printf("Encoder: %ld | Speed: %d\n", currentPos[1], driveSpeed);
          encoder[1].pos = 0;
          resetMsg();
        }
}

void resetMsg(){
  side = "none";
  pred = "idle";
}

void idle(){
        setMotor(0, 0, IN1Pin[0], IN2Pin[0]);
        setMotor(0, 0, IN1Pin[1], IN2Pin[1]);
}

// Hardware functions
void setMotor(int dir, int pwm, int in1, int in2) {
  if (dir == 1) { 
    ledcWrite(in1, pwm); ledcWrite(in2, 0); }
  else if (dir == -1) { 
    ledcWrite(in1, 0); ledcWrite(in2, pwm); }
  else { 
    ledcWrite(in1, 0); ledcWrite(in2, 0); }
}

void ARDUINO_ISR_ATTR encoderISR(void* arg) {
  Encoder* s = static_cast<Encoder*>(arg);
  int b = digitalRead(s->PinB);
  if (b > 0) s->pos++;
  else s->pos--;
}



void ARDUINO_ISR_ATTR buttonISR(void* arg) {
  Button* s = static_cast<Button*>(arg); 
  uint32_t pressTime = millis(); 
  if (pressTime > s->nextPressTime) { 
    s->numberPresses += 1; 
    s->pressed = true; 
    s->nextPressTime = pressTime + cDebounceDelay; 
  }  
}