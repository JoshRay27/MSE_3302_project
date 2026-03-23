import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # First convolution layer
        # Takes a 1-channel image (grayscale) and learns 16 filters
        # Each filter is 3x3 and padding=1 keeps the output the same
        
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # Second convolution layer:
        # Takes the 16 feature maps from conv1 and learns 32 new filter
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # Adaptive Average Pooling:
        # No matter what the input image size is
        # Output will always be 32×7×7 regardless of input size
        # This makes the network input-size independent.
        self.gap = nn.AdaptiveAvgPool2d((7, 7))

        # First fully-connected layer:
        # Takes the flattened 32x7x7 features and maps them to 128 neurons.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)

        # Final Classification Layer:
        # Maps the 128 features to the number of utput classes.
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply first convolution + ReLU activation
        x = F.relu(self.conv1(x))

        # Downsample using max pooling (reduces H and W by 2x)
        x = F.max_pool2d(x, 2)

        # Apply second convolution _ ReLU
        x = F.relu(self.conv2(x))

        # Downsample Again
        x = F.max_pool2d(x, 2)

        # Force feature map to be exactly 32x7x7
        x = self.gap(x)  # <--- makes model input‑size independent

        # Flatten the tensor into a single vector per image
        x = x.view(x.size(0), -1)

        # Fully-connected layer + ReLU
        x = F.relu(self.fc1(x))

        # Final output layer (logits)
        return self.fc2(x)

"""
    Take-Aways
    
    Convolution Layers: These act like feature detectors, they learn to recognize edges, textures, shapes etc.
    Pooling: This reduces image siez and keeps only the most important information.
    Adaptive pooling: This makes the CNN input-size independent. No matter the size of image the network always ends up with a fixed size feature map
                        this will make adding photos in the future easy
    Fully-connected layers: takes the extracted features and makes the final classification decision.
"""