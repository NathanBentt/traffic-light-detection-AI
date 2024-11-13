"""
Manages data loading and preprocessing for training, validation, and testing datasets.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders(trainDir=None, valDir=None, testDir=None, batchSize=32, num_workers=8):
    # Data augmentation for training dataset
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    # Transforms for validation and test datasets (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    trainLoader = valLoader = testLoader = None

    if trainDir:
        trainDataset = datasets.ImageFolder(trainDir, transform=train_transform)
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=num_workers,
                                 pin_memory=True)

    if valDir:
        valDataset = datasets.ImageFolder(valDir, transform=val_test_transform)
        valLoader = DataLoader(valDataset, batch_size=batchSize, shuffle=False, num_workers=num_workers,
                               pin_memory=True)

    if testDir:
        testDataset = datasets.ImageFolder(testDir, transform=val_test_transform)
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False, num_workers=num_workers,
                                pin_memory=True)
        
    # 3 data loaders because we have 3 datasets (train, validation, test)

    return trainLoader, valLoader, testLoader
