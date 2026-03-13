import os
import shutil
import random
from pathlib import Path

#fixing random seed for reproducibility
SEED = 42
random.seed(SEED)

#path to the raw dataset and the directory for processed data
RAW_DATA = "data/raw/asl_alphabet_train/asl_alphabet_train"
PROCESSED_DATA = "data/processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_dataset():
    if os.path.exists(PROCESSED_DATA):
        shutil.rmtree(PROCESSED_DATA)

    os.makedirs(os.path.join(PROCESSED_DATA, "train"))
    os.makedirs(os.path.join(PROCESSED_DATA, "val"))
    os.makedirs(os.path.join(PROCESSED_DATA, "test"))

    classes = [d for d in os.listdir(RAW_DATA) if os.path.isdir(os.path.join(RAW_DATA, d))]

    for class_name in classes:

        class_path = os.path.join(RAW_DATA, class_name)

        images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]

        random.shuffle(images)

        total = len(images)

        train_split = int(TRAIN_RATIO * total)
        val_split = int((TRAIN_RATIO + VAL_RATIO) * total)

        train_imgs = images[:train_split]
        val_imgs = images[train_split:val_split]
        test_imgs = images[val_split:]

        for split, img_list in zip(
            ["train", "val", "test"],
            [train_imgs, val_imgs, test_imgs]
        ):

            dest_dir = os.path.join(PROCESSED_DATA, split, class_name)
            Path(dest_dir).mkdir(parents=True, exist_ok=True)

            for img in img_list:

                src = os.path.join(class_path, img)
                dst = os.path.join(dest_dir, img)

                shutil.copy(src, dst)

    print("Dataset successfully split!")


if __name__ == "__main__":
    split_dataset()