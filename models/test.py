"""
Evaluates the trained model on the test data.
"""

import torch
from torch import nn
from data_loader import get_dataloaders
from model import get_model
import os


def main():
    # Path to test data
    testDir = r'C:\Users\jnb20\Desktop\Code\Datasets\processed\traffic_light_detection\tests'

    batchSize = 64
    num_workers = 8

    # GPU configuration if using CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, testLoader = get_dataloaders(None, None, testDir, batchSize, num_workers)

    model = get_model(weights=None, numClasses=2)  # weights=None because we are loading custom trained weights
    model.load_state_dict(torch.load('models/traffic_light_model.pth', weights_only=True))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    evaluate_model(model, testLoader, criterion, device, mode='Test')


def evaluate_model(model, dataLoader, criterion, device, mode='Validation'):
    model.eval()  # Set model to evaluation mode
    totalLoss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            totalLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avgLoss = totalLoss / len(dataLoader)
    accuracy = 100 * correct / total
    print(f'{mode} Loss: {avgLoss:.4f}, {mode} Accuracy: {accuracy:.2f}%')


if __name__ == "__main__":
    main()
