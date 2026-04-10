"""
wlasl_evaluate.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from wlasl_dataloader import get_wlasl_dataloader
from utils import (
    top_k_accuracy,
    prediction_entropy,
    frame_entropy_stats,
    print_results_table,
    save_json,
    save_frame_predictions,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR  = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

FRAME_PREDICTIONS_FILE = os.path.join(RESULTS_DIR, "frame_predictions.pt")

BEST_MODEL_PATH  = os.path.join(MODELS_DIR, "wlasl_word_model_best.pth")
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, "wlasl_word_model.pth")


class VideoWordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)

        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B, T, 512)
        video_features = self.dropout(features.mean(dim=1))
        return self.classifier(video_features)

    def forward_per_frame(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.feature_extractor(x).view(B * T, 512)
        logits   = self.classifier(features)
        return logits.view(B, T, -1)


def main():
    dataloader, classes = get_wlasl_dataloader(batch_size=4, split="test")
    num_classes = len(classes)

    model = VideoWordClassifier(num_classes).to(device)

    model_path = BEST_MODEL_PATH if os.path.exists(BEST_MODEL_PATH) else FINAL_MODEL_PATH
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    all_preds        = []
    all_labels       = []
    all_logits_list  = []
    frame_sequences  = []
    sample_entropies = []

    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)

            video_logits = model(frames)
            preds        = torch.argmax(video_logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_logits_list.append(video_logits.cpu())

            frame_logits = model.forward_per_frame(frames)

            for b in range(frames.size(0)):
                clip_logits = frame_logits[b].cpu()
                clip_probs  = F.softmax(clip_logits, dim=1)
                true_label  = labels[b].item()

                frame_sequences.append({"logits": clip_logits, "label": true_label})

                stats = frame_entropy_stats(clip_probs)
                sample_entropies.append(stats["mean_entropy"])

    all_logits   = torch.cat(all_logits_list, dim=0)
    all_probs    = F.softmax(all_logits, dim=1)
    label_tensor = torch.tensor(all_labels)

    top1 = accuracy_score(all_labels, all_preds)
    top3 = top_k_accuracy(all_probs, label_tensor, k=3)
    top5 = top_k_accuracy(all_probs, label_tensor, k=5)

    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=classes, zero_division=0
    )

    mean_entropy = sum(sample_entropies) / len(sample_entropies)
    max_entropy  = max(sample_entropies)
    min_entropy  = min(sample_entropies)

    print_results_table(
        {
            "Top-1 Accuracy":         top1,
            "Top-3 Accuracy":         top3,
            "Top-5 Accuracy":         top5,
            "Mean Per-Frame Entropy": mean_entropy,
            "Max Per-Frame Entropy":  max_entropy,
            "Min Per-Frame Entropy":  min_entropy,
            "Num Samples":            len(all_labels),
            "Num Classes":            num_classes,
        },
        title="WLASL Evaluation",
    )
    print("\nClassification Report:\n")
    print(report)

    with open(os.path.join(RESULTS_DIR, "wlasl_test_accuracy.txt"), "w") as f:
        f.write(f"Top-1 Accuracy : {top1 * 100:.2f}%\n")
        f.write(f"Top-3 Accuracy : {top3 * 100:.2f}%\n")
        f.write(f"Top-5 Accuracy : {top5 * 100:.2f}%\n")

    with open(os.path.join(RESULTS_DIR, "wlasl_confusion_matrix.txt"), "w") as f:
        f.write(str(cm))

    with open(os.path.join(RESULTS_DIR, "wlasl_classification_report.txt"), "w") as f:
        f.write(report)

    save_json(
        {
            "mean_per_frame_entropy": mean_entropy,
            "max_per_frame_entropy":  max_entropy,
            "min_per_frame_entropy":  min_entropy,
            "per_sample_mean_entropies": sample_entropies,
        },
        os.path.join(RESULTS_DIR, "uncertainty_summary.json"),
    )

    save_frame_predictions(frame_sequences, FRAME_PREDICTIONS_FILE)
    print("\nAll evaluation files saved to results/")


if __name__ == "__main__":
    main()