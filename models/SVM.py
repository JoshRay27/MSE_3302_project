import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F

class SVMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # single fullt-connected (linear) Layer:
        # - input_dim is the flattened size of the image (e.g., 128 * 128)
        # - num_classes is the number of output categories
        
        # this layer computes: output = xW^T + b
        # It acts like a linear SVM decision funtion. 
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten the input image:
        # Converts (batch, channels, height, width)
        # into (batch, input_dim)
        # Example: (1, 1, 128, 128) -> (1, 16384)
        x = x.view(x.size(0), -1)

        # Pass the flattened vector through the linear layer
        # This produces raw class scores ("logits")
        return self.linear(x)

    
    def hinge_loss(self, outputs, labels):
        # multi-class hinge loss implementation
        # Convert labels into hot-one vectors
        # Example: label 3 -> [0,0,0,1,0,0,0,0,0,0]
        one_hot = F.one_hot(labels, num_classes=outputs.size(1)).float()

        # Compute margins for each class:
        # margin = 1 - (score_of_correct_class)
        # for incorrect classes, this encourages their scores to be lower
        margins = 1 - outputs * one_hot
        return torch.clamp(margins, min=0).mean()
    
    """
    Linear Classifier: There are no convolution layers, no hidden layers, and no non-linear activations

    behaves like a simple SVM but instead of solving a quadratic optimization problem (like scikit-learn),
    it uses gradient descent and integrates with PyTorch

    It flattens the images

    It uses hinge Loss: this encourages the correct class score to be at least 1 higher than the incorrect class scores

    It works with trainning.py because it inherits nn.Module, has .parameters() and uses .forward() methods
    """