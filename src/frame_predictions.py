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
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)              
        features = features.view(B, T, 512)              

        outputs = self.classifier(features)              
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

            logits = model(frames)          
            logits = logits.squeeze(0).cpu() 

            all_sequences.append({
                "logits": logits,
                "label": label.item()
            })

    os.makedirs("results", exist_ok=True)
    torch.save(all_sequences, OUTPUT_FILE)

    print(f"Saved frame-level predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()