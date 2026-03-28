import os
import torch
import torch.nn as nn
from torchvision import models
from wlasl_dataloader import get_wlasl_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/wlasl_word_model.pth"
OUTPUT_FILE = "results/frame_predictions.pt"


class VideoWordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)              # (B*T, 512, 1, 1)
        features = features.view(B, T, 512)              # (B, T, 512)

        # frame-level logits
        outputs = self.classifier(features)              # (B, T, num_classes)
        return outputs


def main():
    dataloader, classes = get_wlasl_dataloader(batch_size=1)

    model = VideoWordClassifier(len(classes)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_sequences = []

    with torch.no_grad():
        for frames, label in dataloader:
            frames = frames.to(device)

            logits = model(frames)           # (1, T, num_classes)
            logits = logits.squeeze(0).cpu() # (T, num_classes)

            all_sequences.append({
                "logits": logits,
                "label": label.item()
            })

    os.makedirs("results", exist_ok=True)
    torch.save(all_sequences, OUTPUT_FILE)

    print(f"Saved frame-level predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()