import os
import json
import random
import torch

INPUT_FILE      = "results/frame_predictions.pt"
OUTPUT_FILE     = "results/sequences.pt"
TEMPLATES_FILE  = "results/sequence_templates.json"
SEQ_LENGTH      = 3
NUM_SEQS        = 300
SEED            = 42

TEMPLATES = [
    #"who is that man"
    [9, 4, 8],   #who, man, shirt
    [9, 4, 7],   #who, man, play
    [9, 5, 4],   #who, many, man
    #"play basketball"
    [4, 7, 0],   #man, play, basketball
    [9, 7, 0],   #who, play, basketball
    [5, 7, 0],   #many, play, basketball
    #"city birthday"
    [3, 1, 6],   #city, birthday, orange
    [3, 4, 2],   #city, man, but
    [3, 9, 4],   #city, who, man
    #"orange shirt"
    [6, 8, 4],   #orange, shirt, man
    [1, 6, 8],   #birthday, orange, shirt
    [2, 6, 8],   #but, orange, shirt
]

TEMPLATE_WEIGHTS = [3, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2]


def build_sequences(samples, num_seqs=NUM_SEQS, seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)

    class_to_samples = {}
    for s in samples:
        lbl = s["label"]
        class_to_samples.setdefault(lbl, []).append(s)

    available_classes = set(class_to_samples.keys())

    valid_templates = [
        t for t in TEMPLATES
        if all(c in available_classes for c in t)
    ]

    if not valid_templates:
        print("Warning: no valid templates found, falling back to random sampling.")
        classes = list(available_classes)
        valid_templates = [
            random.sample(classes, SEQ_LENGTH)
            for _ in range(len(TEMPLATES))
        ]
        weights = None
    else:
        weights = TEMPLATE_WEIGHTS[:len(valid_templates)]

    sequences = []
    template_usage = []

    for _ in range(num_seqs):
        template = random.choices(valid_templates, weights=weights, k=1)[0]
        template_usage.append(template)

        clip_logits = []
        clip_labels = []
        for c in template:
            clip = random.choice(class_to_samples[c])
            clip_logits.append(clip["logits"])
            clip_labels.append(clip["label"])

        sequences.append({
            "clip_logits": clip_logits,
            "clip_labels": clip_labels,
            "seq_label":   clip_labels,
        })

    return sequences, template_usage


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}.\n"
            "Run wlasl_evaluate.py first."
        )

    samples = torch.load(INPUT_FILE)
    print(f"Loaded {len(samples)} clip samples.")

    sequences, template_usage = build_sequences(samples)
    print(f"Built {len(sequences)} sequences of length {SEQ_LENGTH}.")

    from collections import Counter
    bigram_counts = Counter()
    for t in template_usage:
        for i in range(len(t) - 1):
            bigram_counts[(t[i], t[i+1])] += 1

    print(f"\nTop bigrams in sequences (gives HMM something to learn):")
    for (a, b), cnt in bigram_counts.most_common(5):
        print(f"  class {a} → class {b} : {cnt} times")

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    torch.save(sequences, OUTPUT_FILE)
    print(f"\nSaved → {OUTPUT_FILE}")

    with open(TEMPLATES_FILE, "w") as f:
        json.dump({
            "templates": TEMPLATES,
            "template_weights": TEMPLATE_WEIGHTS,
            "bigram_counts": {f"{a}->{b}": cnt for (a,b), cnt in bigram_counts.items()},
        }, f, indent=4)
    print(f"Saved → {TEMPLATES_FILE}")

    seq = sequences[0]
    print(f"\nSample sequence 0:")
    print(f"  Word labels : {seq['clip_labels']}")
    print(f"  Clip shapes : {[t.shape for t in seq['clip_logits']]}")


if __name__ == "__main__":
    main()