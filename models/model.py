"""
Defines the neural network architcture for the model. Used ResNet-18 model with a custom fully connected layer.
"""

import torch.nn as nn
from torchvision import models


def get_model(weights=None, numClasses=2, dropoutRate=0.5):
    # Using ResNet-18 model
    model = models.resnet18(weights=weights)

    numFeatures = model.fc.in_features

    # Replace the final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(dropoutRate),
        nn.Linear(numFeatures, numClasses)
    )

    return model
