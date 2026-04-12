import io
import os
import base64
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

MODEL_PATH     = os.path.join("models", "wlasl_word_model_best.pth")
SEQUENCES_PATH = os.path.join("results", "sequences.pt")

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

TEMPERATURE = 0.5
CLASSES = [
    "basketball", "birthday", "but", "city", "man",
    "many", "orange", "play", "shirt", "who",
]


def build_uniform_transition(num_classes: int, self_loop: float = 0.1):
    off = (1.0 - self_loop) / max(num_classes - 1, 1)
    trans = torch.full((num_classes, num_classes), off)
    trans.fill_diagonal_(self_loop)
    return torch.log(trans.clamp(min=1e-12))


def estimate_transition(sequences, num_classes: int):
    counts = torch.ones(num_classes, num_classes)
    for seq in sequences:
        labels = seq["clip_labels"]
        for i in range(len(labels) - 1):
            counts[labels[i], labels[i + 1]] += 1.0
    trans = counts / counts.sum(dim=1, keepdim=True)
    return torch.log(trans.clamp(min=1e-12))


def clip_log_emission(clip_logits: torch.Tensor, temperature: float = TEMPERATURE):
    probs = F.softmax(clip_logits / temperature, dim=1).mean(dim=0)
    return torch.log(probs.clamp(min=1e-12))


def viterbi(log_emissions, log_trans):
    T, N = len(log_emissions), log_emissions[0].shape[0]
    log_prior = torch.full((N,), -math.log(N))
    dp      = torch.zeros(T, N)
    backptr = torch.zeros(T, N, dtype=torch.long)
    dp[0]   = log_prior + log_emissions[0]
    for t in range(1, T):
        scores         = dp[t - 1].unsqueeze(1) + log_trans
        best_scores, best_states = scores.max(dim=0)
        dp[t]      = best_scores + log_emissions[t]
        backptr[t] = best_states
    best = dp[-1].argmax().item()
    path = [best]
    for t in range(T - 1, 0, -1):
        best = backptr[t, best].item()
        path.append(best)
    path.reverse()
    return path


def prediction_entropy(probs: torch.Tensor) -> float:
    p = probs.clamp(min=1e-12)
    return float(-torch.sum(p * torch.log(p)))

app     = Flask(__name__)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = None
log_trans = None

frame_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_model():
    global model, log_trans
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run wlasl_train.py first.")

    num_classes = len(CLASSES)
    model = VideoWordClassifier(num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    if os.path.exists(SEQUENCES_PATH):
        sequences = torch.load(SEQUENCES_PATH)
        log_trans = estimate_transition(sequences, num_classes)
        app.logger.info("Loaded estimated transition matrix.")
    else:
        log_trans = build_uniform_transition(num_classes)
        app.logger.info("Using uniform transition matrix (sequences.pt not found).")

    app.logger.info(f"Model loaded on {device}. Classes: {CLASSES}")


def decode_frames(b64_list: list) -> torch.Tensor:
    frames = []
    for b64 in b64_list:
        raw  = base64.b64decode(b64)
        img  = Image.open(io.BytesIO(raw)).convert("RGB")
        frames.append(frame_transform(img))
    if not frames:
        raise ValueError("frames list is empty")
    return torch.stack(frames)          


def infer_clip(frames_tensor: torch.Tensor) -> dict:
    T, C, H, W = frames_tensor.shape
    x = frames_tensor.unsqueeze(0).to(device)          
    with torch.no_grad():
        logits = model(x)                              
    probs     = F.softmax(logits, dim=1)               
    avg_probs = probs.mean(dim=0)                      
    pred_idx  = avg_probs.argmax().item()
    top3_idx  = avg_probs.topk(3).indices.tolist()
    return {
        "logits":    logits.cpu(),
        "avg_probs": avg_probs,
        "pred_idx":  pred_idx,
        "top3_idx":  top3_idx,
        "entropy":   prediction_entropy(avg_probs),
    }

@app.get("/health")
def health():
    return jsonify({
        "status":  "ok",
        "device":  str(device),
        "classes": CLASSES,
        "model":   MODEL_PATH,
    })


@app.post("/predict/clip")
def predict_clip():
    body = request.get_json(silent=True)
    if not body or "frames" not in body:
        return jsonify({"error": "JSON body must contain a 'frames' key"}), 400

    try:
        frames_tensor = decode_frames(body["frames"])
    except Exception as e:
        return jsonify({"error": f"Frame decoding failed: {e}"}), 422

    result   = infer_clip(frames_tensor)
    pred_idx = result["pred_idx"]
    probs    = result["avg_probs"]

    return jsonify({
        "prediction":  CLASSES[pred_idx],
        "confidence":  round(probs[pred_idx].item(), 4),
        "entropy":     round(result["entropy"], 4),
        "top3": [
            {"label": CLASSES[i], "probability": round(probs[i].item(), 4)}
            for i in result["top3_idx"]
        ],
    })


@app.post("/predict/sequence")
def predict_sequence():
    body = request.get_json(silent=True)
    if not body or "clips" not in body:
        return jsonify({"error": "JSON body must contain a 'clips' key"}), 400
    if len(body["clips"]) < 2:
        return jsonify({"error": "At least 2 clips required for sequence decoding"}), 400

    log_emissions = []
    slot_details  = []

    for clip_b64_list in body["clips"]:
        try:
            frames_tensor = decode_frames(clip_b64_list)
        except Exception as e:
            return jsonify({"error": f"Frame decoding failed: {e}"}), 422

        result    = infer_clip(frames_tensor)
        log_em    = clip_log_emission(result["logits"])
        log_emissions.append(log_em)

        pred_idx  = result["pred_idx"]
        probs     = result["avg_probs"]
        slot_details.append({
            "greedy":     CLASSES[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "entropy":    round(result["entropy"], 4),
        })

    hmm_path = viterbi(log_emissions, log_trans)
    for i, slot in enumerate(slot_details):
        slot["hmm"] = CLASSES[hmm_path[i]]

    mean_entropy = round(sum(s["entropy"] for s in slot_details) / len(slot_details), 4)

    return jsonify({
        "sequence":     [CLASSES[i] for i in hmm_path],
        "slots":        slot_details,
        "mean_entropy": mean_entropy,
    })

@app.post("/predict/logits")
def predict_logits():
    body = request.get_json(silent=True)
    if not body or "logits" not in body:
        return jsonify({"error": "JSON body must contain a 'logits' key"}), 400

    try:
        logits = torch.tensor(body["logits"], dtype=torch.float32)   
        if logits.dim() != 2:
            raise ValueError("logits must be a 2-D array (T x num_classes)")
    except Exception as e:
        return jsonify({"error": f"logits decoding failed: {e}"}), 422

    probs    = F.softmax(logits, dim=1)
    avg_prob = probs.mean(dim=0)                       
    pred_idx = avg_prob.argmax().item()
    top3_idx = avg_prob.topk(3).indices.tolist()

    return jsonify({
        "prediction": CLASSES[pred_idx],
        "confidence": round(avg_prob[pred_idx].item(), 4),
        "entropy":    round(prediction_entropy(avg_prob), 4),
        "top3": [
            {"label": CLASSES[i], "probability": round(avg_prob[i].item(), 4)}
            for i in top3_idx
        ],
    })


@app.post("/predict/sequence/logits")
def predict_sequence_logits():
    body = request.get_json(silent=True)
    if not body or "clips_logits" not in body:
        return jsonify({"error": "JSON body must contain a 'clips_logits' key"}), 400
    if len(body["clips_logits"]) < 2:
        return jsonify({"error": "At least 2 clips required"}), 400

    log_emissions = []
    slot_details  = []

    for raw in body["clips_logits"]:
        try:
            logits = torch.tensor(raw, dtype=torch.float32)  
        except Exception as e:
            return jsonify({"error": f"logits decoding failed: {e}"}), 422

        log_em   = clip_log_emission(logits)
        log_emissions.append(log_em)

        probs    = F.softmax(logits, dim=1)
        avg_prob = probs.mean(dim=0)
        pred_idx = avg_prob.argmax().item()
        slot_details.append({
            "greedy":     CLASSES[pred_idx],
            "confidence": round(avg_prob[pred_idx].item(), 4),
            "entropy":    round(prediction_entropy(avg_prob), 4),
        })

    hmm_path = viterbi(log_emissions, log_trans)
    for i, slot in enumerate(slot_details):
        slot["hmm"] = CLASSES[hmm_path[i]]

    mean_entropy = round(sum(s["entropy"] for s in slot_details) / len(slot_details), 4)

    return jsonify({
        "sequence":     [CLASSES[i] for i in hmm_path],
        "slots":        slot_details,
        "mean_entropy": mean_entropy,
    })

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)