import cv2
import torch
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from visionPreprocess import preprocess_with_yolo
from ultralytics import YOLO
import serial
import time

# -----------------------------
# Serial (Jetson Nano → ESP32)
# -----------------------------
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(2)  # allow ESP32 to reboot

# -----------------------------
# Model + YOLO
# -----------------------------
from train import NUM_CLASSES
MODEL_PATH = "simple_cnn_model.pth"
yolo = YOLO("yolov8n.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Jetson Nano GStreamer pipeline
# -----------------------------
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():

    # -----------------------------
    # Choose camera source
    # -----------------------------
    # For CSI camera:
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    # For USB webcam instead:
    # cap = cv2.VideoCapture(0)

    print("Camera opened:", cap.isOpened())
    if not cap.isOpened():
        print("Camera failed to open")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed")
                break

            H, W, _ = frame.shape

            # -----------------------------
            # Define left/right ROIs
            # -----------------------------
            x1 = 0
            y1 = 0
            x2 = W // 2
            y2 = H
            x3 = W

            roi_left  = frame[y1:y2, x1:x2]
            roi_right = frame[y1:y2, x2:x3]   # FIXED

            # -----------------------------
            # YOLO preprocessing
            # -----------------------------
            tensor_left  = preprocess_with_yolo(roi_left, yolo)
            tensor_right = preprocess_with_yolo(roi_right, yolo)

            tensors = [
                ("left", tensor_left),
                ("right", tensor_right)
            ]

            # -----------------------------
            # Run predictions
            # -----------------------------
            for side, ten in tensors:
                if ten is not None:
                    ten = ten.unsqueeze(0).to(device)

                    with torch.no_grad():
                        output = model(ten)
                        pred = torch.argmax(output, dim=1).item()

                    # -----------------------------
                    # Send prediction to ESP32
                    # -----------------------------
                    message = f"{side}:{pred}\n"
                    ser.write(message.encode())

                    # -----------------------------
                    # Draw prediction on screen
                    # -----------------------------
                    if side == "left":
                        cv2.putText(frame, f"Left: {pred}", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    else:
                        cv2.putText(frame, f"Right: {pred}", (W - 250, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # -----------------------------
            # Draw ROI divider line
            # -----------------------------
            cv2.line(frame, (x2, 0), (x2, H), (255, 0, 0), 2)

            # -----------------------------
            # Display
            # -----------------------------
            cv2.imshow("Camera", frame)
            print(f"Prediction: {pred}")

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()