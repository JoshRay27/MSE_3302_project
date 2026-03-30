import cv2
import torch
from ultralytics import YOLO
from visionPreprocess import preprocess_with_yolo
from models.model_CNN import SimpleCNN
from train import NUM_CLASSES

# Load YOLO once
yolo = YOLO("yolov8n.pt")

# Load CNN once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("simple_cnn_model.pth", map_location=device))
model.to(device)
model.eval()


def predict_single_frame():
    """Capture one frame, run YOLO preprocessing, run CNN, return prediction."""

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera failed to open")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame")

    H, W, _ = frame.shape

    # Split into left and right ROIs
    roi_left  = frame[:, :W//2]
    roi_right = frame[:, W//2:]

    results = {}

    for side, roi in [("left", roi_left), ("right", roi_right)]:
        tensor = preprocess_with_yolo(roi, yolo)

        if tensor is None:
            results[side] = None
            continue

        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()

        results[side] = pred

    return results