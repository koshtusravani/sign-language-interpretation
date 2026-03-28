import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
from torchvision import models, transforms

MODEL_PATH = "models/cnn_baseline.pth"
CLASS_NAMES_PATH = "data/metadata/class_names.txt"
LEARNED_TRANSITION_PATH = "results/learned_transition_matrix.npy"

WINDOW_NAME = "Live HMM Sign Recognition"

BUFFER_SIZE = 12
CONFIDENCE_THRESHOLD = 0.75
PADDING = 40
STAY_PROB = 0.8
TOP_K = 3


def load_class_names(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing class names file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    if not class_names:
        raise ValueError("No class names found in class_names.txt")

    return class_names


def build_model(num_classes, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}\n"
            "Run cnn_training.py first."
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


def build_default_transition_matrix(num_classes, stay_prob=0.8):
    transition = np.full(
        (num_classes, num_classes),
        (1.0 - stay_prob) / (num_classes - 1),
        dtype=np.float64
    )
    np.fill_diagonal(transition, stay_prob)
    return transition


def load_transition_matrix(num_classes):
    if os.path.exists(LEARNED_TRANSITION_PATH):
        transition = np.load(LEARNED_TRANSITION_PATH)
        if transition.shape == (num_classes, num_classes):
            return transition
    return build_default_transition_matrix(num_classes, STAY_PROB)


def viterbi_decode(emissions, transition_matrix):
    """
    emissions: shape (T, N)
    transition_matrix: shape (N, N)
    """
    T, N = emissions.shape

    emissions = np.clip(emissions, 1e-12, 1.0)
    transition_matrix = np.clip(transition_matrix, 1e-12, 1.0)

    log_emissions = np.log(emissions)
    log_transitions = np.log(transition_matrix)

    dp = np.zeros((T, N), dtype=np.float64)
    backpointer = np.zeros((T, N), dtype=np.int32)

    dp[0] = log_emissions[0]

    for t in range(1, T):
        for j in range(N):
            scores = dp[t - 1] + log_transitions[:, j]
            best_prev = np.argmax(scores)
            dp[t, j] = scores[best_prev] + log_emissions[t, j]
            backpointer[t, j] = best_prev

    best_path = np.zeros(T, dtype=np.int32)
    best_path[-1] = np.argmax(dp[-1])

    for t in range(T - 2, -1, -1):
        best_path[t] = backpointer[t + 1, best_path[t + 1]]

    return best_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = load_class_names(CLASS_NAMES_PATH)
    num_classes = len(class_names)

    model = build_model(num_classes, MODEL_PATH, device)
    transform = get_transform()
    transition_matrix = load_transition_matrix(num_classes)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    emission_buffer = deque(maxlen=BUFFER_SIZE)
    raw_label = "No hand"
    hmm_label = "No hand"
    confidence = 0.0
    top3_lines = []

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
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                raw_label = class_names[pred_idx]

                if confidence >= CONFIDENCE_THRESHOLD:
                    emission_buffer.append(probs)
                else:
                    emission_buffer.append(
                        np.ones(num_classes, dtype=np.float64) / num_classes
                    )

                top3_idx = np.argsort(probs)[::-1][:TOP_K]
                top3_lines = [
                    f"{class_names[i]}: {probs[i] * 100:.1f}%"
                    for i in top3_idx
                ]

                if len(emission_buffer) > 0:
                    emissions = np.array(emission_buffer)
                    decoded = viterbi_decode(emissions, transition_matrix)
                    hmm_idx = int(decoded[-1])
                    hmm_label = class_names[hmm_idx]
                else:
                    hmm_label = raw_label

                mp_draw.draw_landmarks(
                    display,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        if not hand_found:
            raw_label = "No hand"
            hmm_label = "No hand"
            confidence = 0.0
            top3_lines = []
            emission_buffer.clear()

        cv2.rectangle(display, (0, 0), (w, 145), (20, 20, 20), -1)

        cv2.putText(
            display,
            f"Raw CNN: {raw_label}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"HMM Decoded: {hmm_label}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Confidence: {confidence * 100:.1f}%",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        y0 = 180
        for line in top3_lines:
            cv2.putText(
                display,
                line,
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )
            y0 += 30

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()