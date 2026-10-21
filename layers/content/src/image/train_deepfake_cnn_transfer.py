# train_cnn_deepfake_fixed.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("NeuraShield — CNN Deepfake Video Classifier")
    print("=" * 60)
    print(f"Device: {device}")

    # Your paths
    FRAMES_DIR = Path("data/video/processed/frames")
    MODEL_SAVE = "models/image/deepfake_cnn_final.pth"

    # Dataset
    real_imgs = [(str(p), 0) for p in (FRAMES_DIR / "real").glob("*.jpg")]
    fake_imgs = [(str(p), 1) for p in (FRAMES_DIR / "fake").glob("*.jpg")]
    all_data = real_imgs + fake_imgs

    paths = [d[0] for d in all_data]
    labels = [d[1] for d in all_data]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=0.20, random_state=42, stratify=labels
    )

    # Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    class DeepfakeDataset(Dataset):
        def __init__(self, paths, labels, transform=None):
            self.paths = paths
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img = cv2.imread(self.paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]

    train_dataset = DeepfakeDataset(train_paths, train_labels, transform)
    val_dataset = DeepfakeDataset(val_paths, val_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 2)
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(6):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/6 completed")

    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"Model saved: {MODEL_SAVE}")