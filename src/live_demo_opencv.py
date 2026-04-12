import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
from torchvision import models, transforms

from wlasl_dataloader import get_wlasl_dataloader

MODEL_PATH   = "models/wlasl_word_model_best.pth"
WINDOW_NAME  = "Live Sign Recognition — Frame-Based CNN"

SMOOTHING_WINDOW     = 12
CONFIDENCE_THRESHOLD = 0.45
PADDING              = 40
TOP_K                = 3

class VideoWordClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=None)
        for param in backbone.parameters():
            param.requires_grad = False
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout    = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x        = x.view(B * T, C, H, W)
        features = self.feature_extractor(x).view(B * T, 512)
        features = self.dropout(features)
        return self.classifier(features)


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def prediction_entropy(probs_np):
    probs = np.clip(probs_np, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def stable_label(pred_buffer):
    if not pred_buffer:
        return "No hand"
    return Counter(pred_buffer).most_common(1)[0][0]


def main():
    _, classes = get_wlasl_dataloader(split="test")
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}\nRun wlasl_train.py first."
        )

    model = VideoWordClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model — {num_classes} classes: {classes}")

    transform = get_transform()

    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        mp_draw  = mp.solutions.drawing_utils
        hands    = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        use_mediapipe = True
    except ImportError:
        print("MediaPipe not installed — using full frame as ROI.")
        use_mediapipe = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    pred_buffer   = deque(maxlen=SMOOTHING_WINDOW)
    display_label = "No hand"
    display_conf  = 0.0
    display_ent   = 0.0
    topk_text     = []

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        display = frame.copy()
        h, w, _ = frame.shape

        hand_found = False
        roi        = None

        if use_mediapipe:
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                hand_found = True
                lms = result.multi_hand_landmarks[0]
                xs  = [lm.x for lm in lms.landmark]
                ys  = [lm.y for lm in lms.landmark]

                x1 = max(0, int(min(xs) * w) - PADDING)
                y1 = max(0, int(min(ys) * h) - PADDING)
                x2 = min(w, int(max(xs) * w) + PADDING)
                y2 = min(h, int(max(ys) * h) + PADDING)

                roi = frame[y1:y2, x1:x2]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_draw.draw_landmarks(display, lms, mp_hands.HAND_CONNECTIONS)
        else:
            hand_found = True
            roi        = frame

        if hand_found and roi is not None and roi.size > 0:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            tensor  = transform(rgb_roi).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(tensor)                         
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred_idx      = int(np.argmax(probs))
            display_conf  = float(probs[pred_idx])
            display_ent   = prediction_entropy(probs)
            raw_label     = classes[pred_idx]

            if display_conf >= CONFIDENCE_THRESHOLD:
                pred_buffer.append(raw_label)
            else:
                pred_buffer.append("Uncertain")

            display_label = stable_label(pred_buffer)

            top_idx   = np.argsort(probs)[::-1][:TOP_K]
            topk_text = [
                f"{classes[i]}: {probs[i] * 100:.1f}%"
                for i in top_idx
            ]

        else:
            pred_buffer.append("No hand")
            display_label = stable_label(pred_buffer)
            display_conf  = 0.0
            display_ent   = 0.0
            topk_text     = []

        cv2.rectangle(display, (0, 0), (w, 130), (20, 20, 20), -1)

        cv2.putText(display, f"Prediction : {display_label}",
                    (20, 38),  cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0),   2)
        cv2.putText(display, f"Confidence : {display_conf * 100:.1f}%",
                    (20, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 255, 255), 2)
        cv2.putText(display, f"Entropy    : {display_ent:.3f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        y0 = 160
        for line in topk_text:
            cv2.putText(display, line, (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
            y0 += 32

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if use_mediapipe:
        hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()