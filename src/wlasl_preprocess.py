import os
import cv2

RAW_DIR = "data/wlasl/raw_videos"
FRAME_DIR = "data/wlasl/frames"

FRAMES_PER_VIDEO = 20


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    step = max(total_frames // FRAMES_PER_VIDEO, 1)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0 and saved < FRAMES_PER_VIDEO:
            frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        count += 1

    cap.release()


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)

    for word in os.listdir(RAW_DIR):
        word_path = os.path.join(RAW_DIR, word)

        if not os.path.isdir(word_path):
            continue

        print(f"Processing word: {word}")

        for video in os.listdir(word_path):
            if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                continue

            video_path = os.path.join(word_path, video)
            video_name = os.path.splitext(video)[0]

            output_folder = os.path.join(FRAME_DIR, word, video_name)
            os.makedirs(output_folder, exist_ok=True)

            extract_frames(video_path, output_folder)

    print("\nFrame extraction completed successfully!")


if __name__ == "__main__":
    main()