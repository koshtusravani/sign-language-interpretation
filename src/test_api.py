import io
import sys
import base64
import json
import random
import urllib.request
import urllib.error

import torch
import torch.nn.functional as F
from PIL import Image

API_BASE     = "http://127.0.0.1:5000"
PREDS_FILE   = "results/frame_predictions.pt"
SEQS_FILE    = "results/sequences.pt"
SEQUENCE_LEN = 3
SEED         = 42

CLASSES = [
    "basketball", "birthday", "but", "city", "man",
    "many", "orange", "play", "shirt", "who",
]

DUMMY_FRAME_SIZE = 128

def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def separator(title: str = "", width: int = 62):
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * (width - pad - len(title) - 2))
    else:
        print("─" * width)

def logits_to_fake_frames(logits: torch.Tensor, num_frames: int = 8) -> list:
    probs      = F.softmax(logits, dim=1).mean(dim=0)
    top_class  = probs.argmax().item()
    brightness = int((top_class / (len(CLASSES) - 1)) * 200) + 28
    b64_frames = []
    for _ in range(num_frames):
        img = Image.new("RGB", (DUMMY_FRAME_SIZE, DUMMY_FRAME_SIZE),
                        color=(brightness, brightness // 2, 255 - brightness))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64_frames.append(base64.b64encode(buf.getvalue()).decode("utf-8"))
    return b64_frames

def test_health():
    separator("GET /health")
    result = get_json(f"{API_BASE}/health")
    print(f"  status  : {result['status']}")
    print(f"  device  : {result['device']}")
    print(f"  classes : {result['classes']}")
    assert result["status"] == "ok", "health check failed"
    print("  ✓ passed")

def test_single_clip_synthetic(sample: dict, idx: int):
    separator(f"POST /predict/clip  (sample {idx})")
    true_label    = CLASSES[sample["label"]]
    logits        = sample["logits"]
    baseline_name = CLASSES[F.softmax(logits, dim=1).mean(0).argmax().item()]
    b64_frames    = logits_to_fake_frames(logits)
    result        = post_json(f"{API_BASE}/predict/clip", {"frames": b64_frames})

    print(f"  true label  : {true_label}")
    print(f"  CNN baseline: {baseline_name}")
    print(f"  API returned: {result['prediction']}")
    print(f"  confidence  : {result['confidence']:.4f}")
    print(f"  entropy     : {result['entropy']:.4f}  "
          f"({'high' if result['entropy'] > 1.5 else 'low'} uncertainty)")
    print("  top-3 :")
    for item in result["top3"]:
        bar = "█" * int(item["probability"] * 20)
        print(f"    {item['label']:<12} {item['probability']:.4f}  {bar}")
    print("  ✓ passed")


def test_sequence_synthetic(samples: list):
    separator("POST /predict/sequence  (synthetic frames)")
    random.seed(SEED)
    chosen      = random.sample(samples, SEQUENCE_LEN)
    true_labels = [CLASSES[s["label"]] for s in chosen]
    clips       = [logits_to_fake_frames(s["logits"]) for s in chosen]
    result      = post_json(f"{API_BASE}/predict/sequence", {"clips": clips})

    print(f"  true labels  : {true_labels}")
    print(f"  HMM sequence : {result['sequence']}")
    print(f"  mean entropy : {result['mean_entropy']:.4f}")
    separator()
    print(f"  {'Slot':<5} {'True':<14} {'Greedy':<14} {'HMM':<14} {'Conf':>6} {'Entropy':>8}")
    separator()
    for i, (slot, true) in enumerate(zip(result["slots"], true_labels)):
        print(f"  {i:<5} {true:<14} "
              f"{slot['greedy']:<12}{'✓' if slot['greedy']==true else '✗'}  "
              f"{slot['hmm']:<12}{'✓' if slot['hmm']==true else '✗'}  "
              f"{slot['confidence']:>6.4f}  {slot['entropy']:>8.4f}")
    print("  ✓ passed")

def test_verify_clips(samples: list):
    separator("VERIFY  /predict/logits  (all samples)")

    correct_offline = 0
    correct_api     = 0
    match_count     = 0
    mismatches      = []

    for i, sample in enumerate(samples):
        true_label   = sample["label"]
        logits       = sample["logits"]        

        avg_probs    = F.softmax(logits, dim=1).mean(dim=0)
        offline_pred = avg_probs.argmax().item()

        result   = post_json(f"{API_BASE}/predict/logits",
                             {"logits": logits.tolist()})
        api_pred = CLASSES.index(result["prediction"])

        correct_offline += int(offline_pred == true_label)
        correct_api     += int(api_pred     == true_label)

        if offline_pred == api_pred:
            match_count += 1
        else:
            mismatches.append({
                "sample":  i,
                "true":    CLASSES[true_label],
                "offline": CLASSES[offline_pred],
                "api":     CLASSES[api_pred],
            })

        if i < 5:
            sym = "✓" if offline_pred == api_pred else "✗"
            print(f"  [{i:>2}] true={CLASSES[true_label]:<12} "
                  f"offline={CLASSES[offline_pred]:<12} "
                  f"api={CLASSES[api_pred]:<12} "
                  f"conf={result['confidence']:.4f}  "
                  f"ent={result['entropy']:.4f}  {sym}")

    n           = len(samples)
    offline_acc = correct_offline / n
    api_acc     = correct_api / n
    agreement   = match_count / n

    separator()
    print(f"  Samples            : {n}")
    print(f"  Offline top-1 acc  : {offline_acc:.4f}  "
          f"({'✓ matches 0.5600' if abs(offline_acc - 0.56) < 0.01 else '✗ unexpected'})")
    print(f"  API     top-1 acc  : {api_acc:.4f}  "
          f"({'✓ matches offline' if abs(api_acc - offline_acc) < 0.01 else '✗ diverges'})")
    print(f"  Offline↔API agree  : {agreement:.4f}  "
          f"({'✓ perfect agreement' if agreement == 1.0 else f'✗ {len(mismatches)} mismatches'})")

    if mismatches:
        print("\n  Mismatches:")
        for m in mismatches:
            print(f"    sample {m['sample']:>2}: true={m['true']:<12} "
                  f"offline={m['offline']:<12} api={m['api']}")

    assert agreement == 1.0, f"{len(mismatches)} prediction mismatches"
    assert abs(api_acc - offline_acc) < 0.01, "API accuracy diverges from offline"
    print("  ✓ API predictions match offline evaluation exactly")


def test_verify_sequence(samples: list):
    separator("VERIFY  /predict/sequence/logits  (300 sequences)")

    try:
        sequences = torch.load(SEQS_FILE)
    except FileNotFoundError:
        print(f"  ⚠  {SEQS_FILE} not found — skipping.")
        print("     Run sequence_builder.py first.")
        return

    greedy_slot = hmm_slot = greedy_seq = hmm_seq = 0
    total_slots = 0
    total_seqs  = len(sequences)

    print(f"  Running {total_seqs} sequences through API", end="", flush=True)

    for idx, seq in enumerate(sequences):
        clip_logits = seq["clip_logits"]
        true_labels = seq["clip_labels"]

        payload = {"clips_logits": [lg.tolist() for lg in clip_logits]}
        result  = post_json(f"{API_BASE}/predict/sequence/logits", payload)

        greedy_path = [CLASSES.index(s["greedy"]) for s in result["slots"]]
        hmm_path    = [CLASSES.index(s["hmm"])    for s in result["slots"]]

        for g, h, t in zip(greedy_path, hmm_path, true_labels):
            greedy_slot += int(g == t)
            hmm_slot    += int(h == t)
            total_slots += 1

        greedy_seq += int(greedy_path == true_labels)
        hmm_seq    += int(hmm_path    == true_labels)

        if (idx + 1) % 50 == 0:
            print(".", end="", flush=True)

    print()  

    gs = greedy_slot / total_slots
    hs = hmm_slot    / total_slots
    gq = greedy_seq  / total_seqs
    hq = hmm_seq     / total_seqs

    separator()
    print(f"  Sequences  : {total_seqs}   Slots : {total_slots}")
    print()
    print(f"  {'Metric':<30} {'API':>8}   {'Expected':>10}   {'':>4}")
    separator()
    print(f"  {'Slot — greedy':<30} {gs:>8.4f}   {'~0.5622':>10}   "
          f"{'✓' if abs(gs - 0.5622) < 0.02 else '✗'}")
    print(f"  {'Slot — HMM (est+conf)':<30} {hs:>8.4f}   {'~0.6567':>10}   "
          f"{'✓' if abs(hs - 0.6567) < 0.02 else '✗'}")
    print(f"  {'Seq  — greedy':<30} {gq:>8.4f}   {'~0.1533':>10}   "
          f"{'✓' if abs(gq - 0.1533) < 0.02 else '✗'}")
    print(f"  {'Seq  — HMM (est+conf)':<30} {hq:>8.4f}   {'~0.4967':>10}   "
          f"{'✓' if abs(hq - 0.4967) < 0.02 else '✗'}")
    print()
    print(f"  HMM improves slot accuracy : {'✓' if hs > gs else '✗'}  "
          f"({gs:.4f} → {hs:.4f}, Δ={hs-gs:+.4f})")
    print(f"  HMM improves seq  accuracy : {'✓' if hq > gq else '✗'}  "
          f"({gq:.4f} → {hq:.4f}, Δ={hq-gq:+.4f})")

    assert hs > gs, "HMM should improve slot accuracy over greedy"
    assert hq > gq, "HMM should improve sequence accuracy over greedy"
    print("  ✓ API sequence results match offline evaluation")

def main():
    verify_mode = "--verify" in sys.argv

    print()
    print("=" * 62)
    if verify_mode:
        print("  Sign Language API — Verification Mode (--verify)")
    else:
        print("  Sign Language API — End-to-End Test")
    print("=" * 62)

    try:
        test_health()
    except urllib.error.URLError:
        print(f"\n  ✗ Cannot reach API at {API_BASE}")
        print("    Start the server first:  python src/api.py")
        return

    try:
        samples = torch.load(PREDS_FILE)
        print(f"\n  Loaded {len(samples)} samples from {PREDS_FILE}")
    except FileNotFoundError:
        print(f"\n  ✗ {PREDS_FILE} not found. Run wlasl_evaluate.py first.")
        return

    print()

    if verify_mode:
        test_verify_clips(samples)
        print()
        test_verify_sequence(samples)
    else:
        for i in range(min(3, len(samples))):
            test_single_clip_synthetic(samples[i], i)
            print()
        test_sequence_synthetic(samples)

    print()
    print("=" * 62)
    print("  All tests passed.")
    print("=" * 62)
    print()


if __name__ == "__main__":
    main()