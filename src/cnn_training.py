import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader import get_dataloaders
import os
import csv
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

train_loader, val_loader, test_loader = get_dataloaders()

num_classes = len(train_loader.dataset.classes)

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_NAME = "ResNet18"

config = {
    "model": MODEL_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam",
    "device": str(device),
    "dataset": "ASL Alphabet"
}

with open(os.path.join(RESULTS_DIR, "training_config.json"), "w") as f:
    json.dump(config, f, indent=4)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

log_path = os.path.join(RESULTS_DIR, "training_log.csv")

with open(log_path, "w", newline="") as log_file:

    writer = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss"])

    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

        writer.writerow([epoch + 1, running_loss])

model_path = os.path.join(MODELS_DIR, "cnn_baseline.pth")

torch.save(model.state_dict(), model_path)

with open(os.path.join(RESULTS_DIR, "training_summary.txt"), "w") as f:
    f.write("CNN Training Completed\n")
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Device: {device}\n")
    f.write(f"Model saved at: {model_path}\n")

print("Training complete!")
print("Model saved to models/")
print("Training logs saved to results/")