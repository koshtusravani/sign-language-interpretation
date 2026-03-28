import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FRAME_DIR = "data/wlasl/frames"
FRAMES_PER_VIDEO = 20


class WLASLDataset(Dataset):
    def __init__(self, frame_dir=FRAME_DIR, transform=None):
        self.frame_dir = frame_dir
        self.transform = transform

        self.word_classes = sorted([
            d for d in os.listdir(frame_dir)
            if os.path.isdir(os.path.join(frame_dir, d))
        ])

        self.class_to_idx = {word: idx for idx, word in enumerate(self.word_classes)}
        self.samples = []

        for word in self.word_classes:
            word_path = os.path.join(frame_dir, word)

            for video_folder in os.listdir(word_path):
                video_path = os.path.join(word_path, video_folder)

                if os.path.isdir(video_path):
                    self.samples.append((video_path, self.class_to_idx[word]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frame_files = sorted([
            f for f in os.listdir(video_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:FRAMES_PER_VIDEO]

        frames = []
        for frame_file in frame_files:
            img_path = os.path.join(video_path, frame_file)
            image = Image.open(img_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        # pad if fewer than FRAMES_PER_VIDEO
        while len(frames) < FRAMES_PER_VIDEO:
            frames.append(torch.zeros_like(frames[0]))

        frames = torch.stack(frames)  # shape: (T, C, H, W)

        return frames, label


def get_wlasl_dataloader(batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = WLASLDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, dataset.word_classes


if __name__ == "__main__":
    dataloader, classes = get_wlasl_dataloader()

    print("Classes:", classes)

    for frames, labels in dataloader:
        print("Frame batch shape:", frames.shape)  # (B, T, C, H, W)
        print("Labels shape:", labels.shape)
        break