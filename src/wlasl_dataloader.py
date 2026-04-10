import os
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

FRAME_DIR        = "data/wlasl/frames"
FRAMES_PER_VIDEO = 20
TRAIN_RATIO      = 0.8
SEED             = 42


class WLASLDataset(Dataset):
    def __init__(self, samples, word_classes, transform=None):
        self.samples      = samples
        self.word_classes = word_classes
        self.transform    = transform

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
            image    = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            frames.append(image)

        while len(frames) < FRAMES_PER_VIDEO:
            frames.append(torch.zeros_like(frames[0]))

        return torch.stack(frames), label


def _collect_samples(frame_dir):
    word_classes = sorted([
        d for d in os.listdir(frame_dir)
        if os.path.isdir(os.path.join(frame_dir, d))
    ])
    class_to_idx = {word: idx for idx, word in enumerate(word_classes)}

    samples = []
    for word in word_classes:
        word_path = os.path.join(frame_dir, word)
        for video_folder in os.listdir(word_path):
            video_path = os.path.join(word_path, video_folder)
            if os.path.isdir(video_path):
                samples.append((video_path, class_to_idx[word]))

    return samples, word_classes


def _split_samples(samples, train_ratio=TRAIN_RATIO, seed=SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)

    train, test = [], []
    for cls_samples in by_class.values():
        cls_samples = cls_samples[:]
        rng.shuffle(cls_samples)
        n_train = max(1, int(len(cls_samples) * train_ratio))
        train.extend(cls_samples[:n_train])
        test.extend(cls_samples[n_train:])

    return train, test


def _train_transform():
    return transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def _test_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_wlasl_dataloader(batch_size=4, split="train", frame_dir=FRAME_DIR):
    """
    Args:
        split: "train" | "test" | "all"
    """
    samples, word_classes = _collect_samples(frame_dir)
    train_samples, test_samples = _split_samples(samples)

    if split == "train":
        chosen    = train_samples
        transform = _train_transform()
    elif split == "test":
        chosen    = test_samples
        transform = _test_transform()
    elif split == "all":
        chosen    = samples
        transform = _test_transform()
    else:
        raise ValueError(f"split must be 'train', 'test', or 'all', got '{split}'")

    dataset    = WLASLDataset(chosen, word_classes, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return dataloader, word_classes


if __name__ == "__main__":
    train_loader, classes = get_wlasl_dataloader(split="train")
    test_loader,  _       = get_wlasl_dataloader(split="test")

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train batches : {len(train_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    for frames, labels in train_loader:
        print("Train batch — frames:", frames.shape, "labels:", labels.shape)
        break
    for frames, labels in test_loader:
        print("Test  batch — frames:", frames.shape, "labels:", labels.shape)
        break