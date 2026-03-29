import serial
import time

ser = serial.Serial('COM3', 115200, timeout=1)
time.sleep(2)  # allow ESP32 to reboot

print("Sending test messages...")

while True:
    msg = input("Enter message to send (e.g., left:5): ")
    ser.write((msg + "\n").encode())
    print("Sent:", msg)