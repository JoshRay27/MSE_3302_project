import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from visionPreprocess import preprocess_live, preprocess_with_yolo
from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

class PreprocessedImageDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = cv2.imread(img_path)

        '''
        # Use live preprocessing for training
        processed, _ , _1 = preprocess_live(img)
        processed = np.array(processed)
        # Ensure shape is (1, H, W)
        if processed.ndim == 2:
            processed = processed.reshape(1, processed.shape[0], processed.shape[1])
        img_tensor = torch.tensor(processed, dtype=torch.float32)
        '''

        img_tensor = preprocess_with_yolo(img, yolo)
        if img_tensor is None:
            return self.__getitem__((idx + 1) % len(self))

        return img_tensor, label