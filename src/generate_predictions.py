import torch
import numpy as np
from torchvision import models
from data_loader import get_dataloaders
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def generate_predictions(model_path="models/cnn_baseline.pth"):

    train_loader, val_loader, test_loader = get_dataloaders()

    num_classes = len(test_loader.dataset.classes)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    np.save(os.path.join(RESULTS_DIR, "cnn_probabilities.npy"), all_probs)

    print("CNN probabilities saved to results/cnn_probabilities.npy")


if __name__ == "__main__":
    generate_predictions()