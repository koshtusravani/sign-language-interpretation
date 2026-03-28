import os
import json
import shutil

SUBSET_FILE = "archive/nslt_100.json"
METADATA_FILE = "archive/WLASL_v0.3.json"
VIDEO_DIR = "archive/videos"
OUTPUT_DIR = "data/wlasl/raw_videos"

MAX_WORDS = 10


def build_video_to_word_map():
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    video_to_word = {}

    for entry in metadata:
        gloss = entry["gloss"].strip().replace(" ", "_").lower()

        for inst in entry["instances"]:
            video_id = str(inst["video_id"])
            video_to_word[video_id] = gloss

    return video_to_word


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(SUBSET_FILE, "r", encoding="utf-8") as f:
        subset_data = json.load(f)

    video_to_word = build_video_to_word_map()

    selected_words = []
    word_to_videos = {}

    for video_id in subset_data.keys():
        video_id = str(video_id)

        if video_id not in video_to_word:
            continue

        word = video_to_word[video_id]

        if word not in word_to_videos:
            if len(selected_words) >= MAX_WORDS:
                continue
            selected_words.append(word)
            word_to_videos[word] = []

        word_to_videos[word].append(video_id)

    for word, video_ids in word_to_videos.items():
        word_folder = os.path.join(OUTPUT_DIR, word)
        os.makedirs(word_folder, exist_ok=True)

        copied = 0
        for video_id in video_ids:
            src = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

            if os.path.exists(src):
                dst = os.path.join(word_folder, f"{video_id}.mp4")
                shutil.copy2(src, dst)
                copied += 1

        print(f"{word}: copied {copied} videos")

    print("\nFinished organizing raw_videos.")


if __name__ == "__main__":
    main()