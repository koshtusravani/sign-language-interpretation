"""
sequence_builder.py — Build multi-word sequences from isolated sign clips.

The proposal says:
  "sign sequences will be constructed from isolated samples, allowing
   systematic comparison between frame-based and context-aware recognition."

Reads:   results/frame_predictions.pt   (from wlasl_evaluate.py)
Writes:  results/sequences.pt
"""

import os
import random
import torch

INPUT_FILE  = "results/frame_predictions.pt"
OUTPUT_FILE = "results/sequences.pt"
SEQ_LENGTH  = 3
NUM_SEQS    = 200
SEED        = 42


def build_sequences(samples, seq_length=SEQ_LENGTH, num_seqs=NUM_SEQS, seed=SEED):
    """
    Randomly chain seq_length clips from different classes into synthetic sequences.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    class_to_samples = {}
    for s in samples:
        lbl = s["label"]
        class_to_samples.setdefault(lbl, []).append(s)

    classes = list(class_to_samples.keys())
    if len(classes) < seq_length:
        classes = classes * (seq_length // len(classes) + 1)

    sequences = []
    for _ in range(num_seqs):
        chosen_classes = random.sample(classes, seq_length)

        clip_logits = []
        clip_labels = []
        for c in chosen_classes:
            clip = random.choice(class_to_samples[c])
            clip_logits.append(clip["logits"])
            clip_labels.append(clip["label"])

        sequences.append({
            "clip_logits": clip_logits,
            "clip_labels": clip_labels,
            "seq_label":   clip_labels,
        })

    return sequences


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}.\n"
            "Run wlasl_evaluate.py first to generate per-frame predictions."
        )

    samples = torch.load(INPUT_FILE)
    print(f"Loaded {len(samples)} clip samples.")

    sequences = build_sequences(samples)
    print(f"Built {len(sequences)} synthetic sequences of length {SEQ_LENGTH}.")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    torch.save(sequences, OUTPUT_FILE)
    print(f"Saved → {OUTPUT_FILE}")

    seq = sequences[0]
    print(f"\nSample sequence 0:")
    print(f"  Word labels : {seq['clip_labels']}")
    print(f"  Clip shapes : {[t.shape for t in seq['clip_logits']]}")


if __name__ == "__main__":
    main()