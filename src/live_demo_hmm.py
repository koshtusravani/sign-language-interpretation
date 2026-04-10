"""
live_demo_hmm.py — Real-time sign recognition with HMM sequence decoding.

Context-aware demo. Shows side-by-side:
  - Raw CNN prediction (frame-based, no context)
  - HMM Viterbi decoded prediction (context-aware)
  - Per-frame entropy (uncertainty before context)
  - Top-3 predictions

The transition matrix is loaded from results/sequences.pt
(estimated from your training sequences).
Press 'q' to quit.
"""

import os
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torchvision import models, transforms

from wlasl_dataloader import get_wlasl_dataloader
from hmm_sequence import estimate_transition_from_sequences

MODEL_PATH     = "models/wlasl_word_model_best.pth"
SEQUENCES_PATH = "results/sequences.pt"
WINDOW_NAME    = "Live Sign Recognition — HMM Context-Aware"

BUFFER_SIZE          = 15
CONFIDENCE_THRESHOLD = 0.45
PADDING              = 40
TOP_K                = 3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Viterbi (numpy — runs every frame on the rolling buffer)
# ---------------------------------------------------------------------------

def viterbi_decode_np(emission_buffer, log_trans_np):
    """
    emission_buffer: list of (num_classes,) numpy arrays — softmax probs
    log_trans_np:    (num_classes, num_classes) numpy array

    Returns: decoded class index for the most recent frame
    """
    T   = len(emission_buffer)
    N   = emission_buffer[0].shape[0]
    eps = 1e-12

    log_em = np.log(np.clip(np.array(emission_buffer), eps, 1.0))  # (T, N)

    dp      = np.zeros((T, N))
    backptr = np.zeros((T, N), dtype=np.int32)

    dp[0] = log_em[0] - math.log(N)   # uniform prior

    for t in range(1, T):
        scores         = dp[t - 1][:, None] + log_trans_np   # (N, N)
        best_prev      = np.argmax(scores, axis=0)            # (N,)
        dp[t]          = scores[best_prev, np.arange(N)] + log_em[t]
        backptr[t]     = best_prev

    best_last = int(np.argmax(dp[-1]))

    path = [best_last]
    for t in range(T - 1, 0, -1):
        best_last = backptr[t, best_last]
        path.append(best_last)

    path.reverse()
    return path[-1]   # return prediction for most recent frame


def main():
    # ── Classes ──────────────────────────────────────────────────────────
    _, classes = get_wlasl_dataloader(split="test")
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ─────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}\nRun wlasl_train.py first."
        )
    model = VideoWordClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Loaded model — {num_classes} classes")

    # ── Transition matrix ─────────────────────────────────────────────────
    if os.path.exists(SEQUENCES_PATH):
        sequences  = torch.load(SEQUENCES_PATH)
        log_trans  = estimate_transition_from_sequences(sequences, num_classes)
        log_trans_np = log_trans.numpy()
        print("Loaded estimated transition matrix from sequences.pt")
    else:
        # Uniform fallback
        off = (1.0 - 0.1) / max(num_classes - 1, 1)
        trans = np.full((num_classes, num_classes), off)
        np.fill_diagonal(trans, 0.1)
        log_trans_np = np.log(np.clip(trans, 1e-12, 1.0))
        print("Using uniform transition matrix (sequences.pt not found)")

    transform = get_transform()

    # ── MediaPipe ─────────────────────────────────────────────────────────
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

    emission_buffer = deque(maxlen=BUFFER_SIZE)
    raw_label   = "No hand"
    hmm_label   = "No hand"
    entropy_val = 0.0
    topk_text   = []

    print("Press 'q' to quit.")
    print("Left panel  = Raw CNN (no context)")
    print("Right panel = HMM Viterbi (context-aware)")

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

            pred_idx    = int(np.argmax(probs))
            confidence  = float(probs[pred_idx])
            entropy_val = prediction_entropy(probs)
            raw_label   = classes[pred_idx]

            # Add to buffer — use uniform if below confidence threshold
            if confidence >= CONFIDENCE_THRESHOLD:
                emission_buffer.append(probs)
            else:
                emission_buffer.append(
                    np.ones(num_classes, dtype=np.float64) / num_classes
                )

            # HMM decode over buffer
            if len(emission_buffer) >= 2:
                hmm_idx   = viterbi_decode_np(list(emission_buffer), log_trans_np)
                hmm_label = classes[hmm_idx]
            else:
                hmm_label = raw_label

            top_idx   = np.argsort(probs)[::-1][:TOP_K]
            topk_text = [
                f"{classes[i]}: {probs[i] * 100:.1f}%"
                for i in top_idx
            ]

        else:
            emission_buffer.clear()
            raw_label   = "No hand"
            hmm_label   = "No hand"
            entropy_val = 0.0
            topk_text   = []

        # ── HUD ─────────────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (w, 150), (20, 20, 20), -1)

        # Left: raw CNN
        cv2.putText(display, "CNN (no context):",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
        cv2.putText(display, raw_label,
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        # Right: HMM
        cv2.putText(display, "HMM (context-aware):",
                    (w // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
        cv2.putText(display, hmm_label,
                    (w // 2, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # Entropy bar
        max_entropy  = math.log(num_classes)
        entropy_norm = min(entropy_val / max_entropy, 1.0)
        bar_w        = int((w - 40) * entropy_norm)
        bar_color    = (
            int(255 * entropy_norm),
            int(255 * (1 - entropy_norm)),
            50,
        )
        cv2.rectangle(display, (20, 115), (20 + bar_w, 138), bar_color, -1)
        cv2.rectangle(display, (20, 115), (w - 20, 138), (100, 100, 100), 1)
        cv2.putText(display, f"Entropy: {entropy_val:.3f} / {max_entropy:.3f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

        # Top-k
        y0 = 170
        for line in topk_text:
            cv2.putText(display, line, (20, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            y0 += 30

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if use_mediapipe:
        hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()