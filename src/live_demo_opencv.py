import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from torchvision import models, transforms

MODEL_PATH = "models/cnn_baseline.pth"
CLASS_NAMES_PATH = "data/metadata/class_names.txt"
WINDOW_NAME = "Live Sign Recognition"

SMOOTHING_WINDOW = 12
CONFIDENCE_THRESHOLD = 0.75
PADDING = 40
TOP_K = 3


def load_class_names(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing class names file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_model(num_classes, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}\nRun cnn_training.py first."
        )

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def preprocess_roi(roi, transform, device):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    return tensor


def stable_label(pred_buffer):
    if not pred_buffer:
        return "No hand"
    return Counter(pred_buffer).most_common(1)[0][0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(CLASS_NAMES_PATH)
    model = build_model(len(class_names), MODEL_PATH, device)
    transform = get_transform()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    pred_buffer = deque(maxlen=SMOOTHING_WINDOW)
    display_label = "No hand"
    display_conf = 0.0
    topk_text = []

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_found = False

        if result.multi_hand_landmarks:
            hand_found = True
            hand_landmarks = result.multi_hand_landmarks[0]

            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x1 = max(0, int(min(xs) * w) - PADDING)
            y1 = max(0, int(min(ys) * h) - PADDING)
            x2 = min(w, int(max(xs) * w) + PADDING)
            y2 = min(h, int(max(ys) * h) + PADDING)

            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                input_tensor = preprocess_roi(roi, transform, device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)

                    conf, pred = torch.max(probs, dim=1)
                    conf = conf.item()
                    pred_idx = pred.item()

                    top_probs, top_indices = torch.topk(probs, k=TOP_K, dim=1)
                    top_probs = top_probs[0].cpu().numpy()
                    top_indices = top_indices[0].cpu().numpy()

                raw_label = class_names[pred_idx]
                display_conf = conf

                if conf >= CONFIDENCE_THRESHOLD:
                    pred_buffer.append(raw_label)
                else:
                    pred_buffer.append("Uncertain")

                display_label = stable_label(pred_buffer)

                topk_text = [
                    f"{class_names[idx]}: {prob * 100:.1f}%"
                    for idx, prob in zip(top_indices, top_probs)
                ]

                # subtle crop rectangle
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_draw.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        if not hand_found:
            pred_buffer.append("No hand")
            display_label = stable_label(pred_buffer)
            display_conf = 0.0
            topk_text = []

        # top banner
        cv2.rectangle(display, (0, 0), (w, 120), (20, 20, 20), -1)

        cv2.putText(
            display,
            f"Prediction: {display_label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Confidence: {display_conf * 100:.1f}%",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 255, 255),
            2
        )

        # show top-k predictions
        y_offset = 150
        for line in topk_text:
            cv2.putText(
                display,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )
            y_offset += 35

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()