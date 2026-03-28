import os
import cv2
import json
import shutil

RAW_DIR = "data/wlasl/raw_videos"
FRAME_DIR = "data/wlasl/frames"
ANNOTATION_FILE = "archive/nslt_100.json"

FRAMES_PER_VIDEO = 12


def load_action_segments():
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = {}
    for video_id, info in data.items():
        action = info.get("action", None)
        if action and len(action) >= 3:
            start_frame = int(action[1])
            end_frame = int(action[2])
            segments[str(video_id)] = (start_frame, end_frame)

    return segments


def extract_segment_frames(video_path, output_folder, start_frame=None, end_frame=None):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return

    # fallback to whole video if segment is invalid
    if start_frame is None or end_frame is None or end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames - 1

    start_frame = max(0, start_frame)
    end_frame = min(total_frames - 1, end_frame)

    segment_length = end_frame - start_frame + 1
    if segment_length <= 0:
        cap.release()
        return

    indices = []
    if segment_length <= FRAMES_PER_VIDEO:
        indices = list(range(start_frame, end_frame + 1))
    else:
        step = segment_length / FRAMES_PER_VIDEO
        indices = [int(start_frame + i * step) for i in range(FRAMES_PER_VIDEO)]

    saved = 0
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_path = os.path.join(output_folder, f"frame_{saved}.jpg")
        cv2.imwrite(frame_path, frame)
        saved += 1

    cap.release()


def main():
    action_segments = load_action_segments()

    # clear old extracted frames so you don't mix old + new
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
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

            start_frame, end_frame = action_segments.get(video_name, (None, None))
            extract_segment_frames(video_path, output_folder, start_frame, end_frame)

    print("\nSegment-based frame extraction completed successfully!")


if __name__ == "__main__":
    main()