import os
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from wlasl_dataloader import get_wlasl_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_NAME = "ResNet18_WLASL"


class VideoWordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)         # (B*T, 512, 1, 1)
        features = features.view(B, T, 512)          # (B, T, 512)

        video_features = features.mean(dim=1)        # average across frames
        outputs = self.classifier(video_features)    # (B, num_classes)

        return outputs


def main():
    dataloader, classes = get_wlasl_dataloader(batch_size=4)
    num_classes = len(classes)

    model = VideoWordClassifier(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # save config
    config = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "num_classes": num_classes,
        "classes": classes,
        "device": str(device)
    }

    with open(os.path.join(RESULTS_DIR, "wlasl_training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    log_path = os.path.join(RESULTS_DIR, "wlasl_training_log.csv")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["epoch", "train_loss"])

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0

            for frames, labels in dataloader:
                frames = frames.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss:.4f}")
            writer.writerow([epoch + 1, running_loss])

    model_path = os.path.join(MODELS_DIR, "wlasl_word_model.pth")
    torch.save(model.state_dict(), model_path)

    with open(os.path.join(RESULTS_DIR, "wlasl_training_summary.txt"), "w") as f:
        f.write("WLASL Word-Level Training Completed\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Classes: {classes}\n")
        f.write(f"Model saved at: {model_path}\n")

    print("\nTraining complete!")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()