"""
This script trains a model to detect traffic lights in images. 
The model is trained using a dataset of images containing traffic lights 
and images without traffic lights.
"""

import torch
from torch import nn, optim
from torchvision.models import ResNet18_Weights
from data_loader import get_dataloaders
from model import get_model
import os


def main():
    # Paths to training and validation images
    trainDir = r'C:\Users\...\training'
    valDir = r'C:\Users\...\validation'

    batchSize = 64
    learningRate = 0.0001
    numEpochs = 10
    numOfWorkers = 8

    # GPU configuration if using CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    trainLoader, valLoader, _ = get_dataloaders(trainDir, valDir, None, batchSize, numOfWorkers)

    # Initialize the model with pretrained weights
    model = get_model(weights=ResNet18_Weights.IMAGENET1K_V1, numClasses=2)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate, weight_decay=0.001)

    num_training_images = len(trainLoader.dataset)

    train_model(model, trainLoader, valLoader, criterion, optimizer, device, numEpochs, num_training_images)

    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/traffic_light_model.pth')


def train_model(model, trainLoader, valLoader, criterion, optimizer, device, numEpochs, num_training_images):
    model.to(device)

    for epoch in range(numEpochs):
        model.train()  # Set model to training mode
        runningLoss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epochLoss = runningLoss / len(trainLoader)
        epochAcc = 100 * correct / total
        print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {epochLoss:.4f}, Accuracy: {epochAcc:.2f}%')

        evaluate_model(model, valLoader, criterion, device, mode='Validation')

    print(f'Training completed. Number of training images: {num_training_images}')


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


if __name__ == '__main__':
    main()
