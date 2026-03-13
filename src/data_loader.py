import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

DATA_DIR = "data/processed"


def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Classes:", train_loader.dataset.classes)