import cv2
import torch
import torchvision.transforms as transforms
from models.model_CNN import SimpleCNN
from models.complex_CNN import ASLNet
from visionPreprocess import preprocess_live, preprocess_with_yolo
from ultralytics import YOLO
from train import NUM_CLASSES
import serial
import time

ser = serial.Serial('dev/ttyUSB0', 115200, timeout=1)
time.sleep(2) #Allow ESP 32 Time to reboot



#NUM_CLASSES = 10
MODEL_PATH = "simple_cnn_model.pth"
yolo = YOLO("yolov8n.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def main():

    # --- USE NORMAL WEBCAM ---
    cap = cv2.VideoCapture(0)

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

            # Define two ROI centered 400×400 ROI
            x1 = 0
            y1 = 0
            x2 = W//2
            y2 = H
            x3 = W
            y3 = H


            roi_left = frame[y1:y2, x1:x2]
            roi_right = frame[y1:y3, x2:x3]
            # Process the ROI (or full frame depending on your pipeline)
            #live, bbox, hand_region = preprocess_live(roi)
            # tensor = torch.from_numpy(processed).float().unsqueeze(0).to(device)
            
            tensor_left =  preprocess_with_yolo(roi_left, yolo)    
            tensor_right =  preprocess_with_yolo(roi_right, yolo) 
            tensor = [
                ("left", tensor_left),
                ("right",  tensor_right)
                ]
            for side, ten in tensor:
                if ten is not None:
                    ten = ten.unsqueeze(0).to(device)   # <--- ADD THIS
                    with torch.no_grad():
                        output = model(ten)
                        pred = torch.argmax(output, dim=1).item()
                    
                    message = f"{side}:{pred}\n"
                    ser.write(message.encode()) 

                    # Draw prediction
                    if side == 'left':
                        cv2.putText(frame, f"Prediction: {pred}", (x1, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"Prediction: {pred}", (x2, y2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw ROI box
            cv2.rectangle(frame, (x2-1, 0), (x2, y2), (255, 0, 0), 2)

            # Draw bounding box inside ROI if found
            #if bbox is not None:
            #    bx, by, bw, bh = bbox
            #    cv2.rectangle(roi, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

            # Show processed image (scaled back to 0–255)
            #cv2.imshow("Processed", processed[0])
            cv2.imshow("Camera", frame)
            #cv2.imshow("Live", live)
            #if hand_region is not None:
            #    cv2.imshow("Hand_region", hand_region)
            #    #cv2.imshow("Gray", gray)
            print(f"Prediction: {pred}")

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

    finally:
        print("Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()