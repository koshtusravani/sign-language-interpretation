import torch
import os
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_loader import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_model(model_path="models/cnn_baseline.pth"):
    train_loader, val_loader, test_loader = get_dataloaders()

    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)

    with open(os.path.join(RESULTS_DIR, "test_accuracy.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n")

    with open(os.path.join(RESULTS_DIR, "confusion_matrix.txt"), "w") as f:
        f.write(str(cm))

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    print("Evaluation results saved to results/")


if __name__ == "__main__":
    evaluate_model()