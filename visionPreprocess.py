import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms

cnn_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # now shape is (1, H, W)
    transforms.Normalize(mean=[0.5], std=[0.5])
])



def preprocess_live(img, size=(128,128), training=False):
    H, W, _ = img.shape

    # --- Skin detection ---
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 20, 70], dtype=np.uint8)
    upper1 = np.array([20, 255, 255], dtype=np.uint8)

    lower2 = np.array([160, 20, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5,5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        blank = np.zeros((1, size[0], size[1]), dtype=np.float32)
        return blank, None, None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Square crop
    side = max(w, h)
    cx = x + w // 2
    cy = y + h // 2

    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(cx + side // 2, W)
    y2 = min(cy + side // 2, H)

    # Padding
    pad = int(0.05 * side)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, W)
    y2 = min(y2 + pad, H)

    hand_region = img[y1:y2, x1:x2]  # <-- debug crop

    gray = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    crop_mask = skin_mask[y1:y2, x1:x2]
    gray[crop_mask == 0] = 0
    digit = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    digit = digit.astype("float32") / 255.0

    return digit.reshape(1, size[0], size[1]), (x1, y1, x2-x1, y2-y1), gray


def preprocess_with_yolo(img, yolo_model):
    """
    Runs YOLO on an image, crops the highest-confidence detection,
    and returns a CNN-ready tensor.
    """
    # Load image (BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO
    results = yolo_model(img_rgb)[0]

    if len(results.boxes) == 0:
        return None

    # Select highest-confidence box
    boxes = results.boxes
    best_idx = torch.argmax(boxes.conf)
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    # Crop
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Convert to grayscale
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # Resize to your CNN input size (128×128 or 64×64)
    crop_gray = cv2.resize(crop_gray, (128, 128))

    # Convert to tensor: (1, H, W)
    crop_tensor = torch.tensor(crop_gray, dtype=torch.float32).unsqueeze(0) / 255.0


    # Apply CNN transforms
    crop_tensor = cnn_transform(crop_tensor)

    return crop_tensor


# Process folder just used for testing
def process_folder(input_folder, output_folder, size=(128,128)):
    """Process all images in a folder and save the results. """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg",".png", ".bmp", ".tiff")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping unreadable file: {filename}")
                continue

            processed = preprocess_live(img, size)

            # convert back to 0-255 for saving
            save_img = (processed * 255).astype("uint8")
            cv2.imwrite(output_path, save_img)

            print(f"Processed: {filename}")

"""
Grayscale simplifies the data

Models require fixed input size

Gaussian Blur removes noise and helps the model focus on structure rather than pixel-level randomness

Normalize makes pixel values small which is better for neural networks

Have to convert back to 0-255 to save with OpenCV
"""
