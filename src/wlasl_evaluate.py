import os
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wlasl_dataloader import get_wlasl_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


class VideoWordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)      # (B*T, 512, 1, 1)
        features = features.view(B, T, 512)       # (B, T, 512)

        video_features = features.mean(dim=1)     # average across frames
        outputs = self.classifier(video_features)

        return outputs


def main():
    dataloader, classes = get_wlasl_dataloader(batch_size=4)
    num_classes = len(classes)

    model = VideoWordClassifier(num_classes).to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, "wlasl_word_model.pth"), map_location=device)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes)

    with open(os.path.join(RESULTS_DIR, "wlasl_test_accuracy.txt"), "w") as f:
        f.write(f"WLASL Test Accuracy: {acc * 100:.2f}%\n")

    with open(os.path.join(RESULTS_DIR, "wlasl_confusion_matrix.txt"), "w") as f:
        f.write(str(cm))

    with open(os.path.join(RESULTS_DIR, "wlasl_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"WLASL Test Accuracy: {acc * 100:.2f}%")
    print("Saved evaluation files to results/")


if __name__ == "__main__":
    main()