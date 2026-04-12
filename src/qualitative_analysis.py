import os
import torch
import torch.nn.functional as F
import math

from hmm_sequence import (
    estimate_transition_from_sequences,
    clip_emission_log_probs,
    clip_emission_confidence_weighted,
    viterbi_decode,
    greedy_decode,
    TEMPERATURE,
)
from wlasl_dataloader import get_wlasl_dataloader

SEQUENCES_FILE = "results/sequences.pt"
OUTPUT_FILE    = "results/qualitative_analysis.txt"
NUM_EXAMPLES   = 10


def entropy(log_probs):
    probs = log_probs.exp()
    probs = probs.clamp(min=1e-12)
    return float(-torch.sum(probs * torch.log(probs)))


def max_entropy(num_classes):
    return math.log(num_classes)


def main():
    if not os.path.exists(SEQUENCES_FILE):
        raise FileNotFoundError(
            f"Missing {SEQUENCES_FILE}.\nRun sequence_builder.py first."
        )

    sequences   = torch.load(SEQUENCES_FILE)
    num_classes = sequences[0]["clip_logits"][0].shape[1]

    _, classes      = get_wlasl_dataloader(split="test")
    log_trans_est   = estimate_transition_from_sequences(sequences, num_classes)
    max_ent         = max_entropy(num_classes)

    cases = {
        "hmm_corrected":   [],   
        "both_correct":    [],   
        "both_wrong":      [],   
        "hmm_degraded":    [],   
    }

    for seq in sequences:
        clip_logits = seq["clip_logits"]
        true_labels = seq["clip_labels"]

        log_em   = [clip_emission_log_probs(lg, TEMPERATURE) for lg in clip_logits]
        g_path   = greedy_decode(log_em)
        h_path, _= viterbi_decode(log_em, log_trans_est)

        for slot, (lg, true, g_pred, h_pred) in enumerate(
            zip(clip_logits, true_labels, g_path, h_path)
        ):
            probs       = F.softmax(lg, dim=1)
            avg_probs   = probs.mean(dim=0)
            ent         = float(-torch.sum(avg_probs * torch.log(avg_probs.clamp(1e-12))))
            ent_norm    = ent / max_ent
            confidence  = float(avg_probs.max())

            example = {
                "true":       true,
                "true_name":  classes[true],
                "cnn_pred":   g_pred,
                "cnn_name":   classes[g_pred],
                "hmm_pred":   h_pred,
                "hmm_name":   classes[h_pred],
                "entropy":    ent,
                "ent_norm":   ent_norm,
                "confidence": confidence,
                "slot":       slot,
            }

            cnn_correct = (g_pred == true)
            hmm_correct = (h_pred == true)

            if not cnn_correct and hmm_correct:
                cases["hmm_corrected"].append(example)
            elif cnn_correct and hmm_correct:
                cases["both_correct"].append(example)
            elif not cnn_correct and not hmm_correct:
                cases["both_wrong"].append(example)
            elif cnn_correct and not hmm_correct:
                cases["hmm_degraded"].append(example)

    for key in cases:
        seen = set()
        unique = []
        for ex in cases[key]:
            fingerprint = (ex["true"], ex["cnn_pred"], ex["hmm_pred"], round(ex["entropy"], 4))
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(ex)
        cases[key] = unique

    cases["hmm_corrected"].sort(key=lambda x: x["entropy"], reverse=True)
    
    cases["both_correct"].sort(key=lambda x: x["confidence"], reverse=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("QUALITATIVE ANALYSIS — UNCERTAINTY RESOLUTION BY CONTEXT\n")
        f.write(f"Max possible entropy (uniform over {num_classes} classes): "
                f"{max_ent:.4f}\n\n")

        f.write("CASE 1 — HMM CORRECTED HIGH-ENTROPY CNN PREDICTIONS\n")
        f.write("These are the most important examples: the CNN was uncertain\n")
        f.write("or wrong, and the HMM used context to find the right answer.\n")

        shown = 0
        for ex in cases["hmm_corrected"][:NUM_EXAMPLES]:
            f.write(f"\nTrue label   : {ex['true_name']} (class {ex['true']})\n")
            f.write(f"CNN predicted: {ex['cnn_name']} \n")
            f.write(f"HMM decoded  : {ex['hmm_name']} \n")
            f.write(f"Entropy      : {ex['entropy']:.4f} / {max_ent:.4f} "
                    f"({ex['ent_norm'] * 100:.1f}% of max)\n")
            f.write(f"Confidence   : {ex['confidence'] * 100:.1f}%\n")
            shown += 1

        if shown == 0:
            f.write("No examples found where HMM corrected CNN errors.\n")

        f.write(f"\n\nCASE 2 — BOTH CORRECT (CNN + HMM AGREE)\n")
        f.write("High-confidence examples where context reinforces the CNN.\n")

        for ex in cases["both_correct"][:5]:
            f.write(f"\nTrue label   : {ex['true_name']}\n")
            f.write(f"CNN predicted: {ex['cnn_name']} \n")
            f.write(f"HMM decoded  : {ex['hmm_name']} \n")
            f.write(f"Entropy      : {ex['entropy']:.4f} ({ex['ent_norm'] * 100:.1f}% of max)\n")
            f.write(f"Confidence   : {ex['confidence'] * 100:.1f}%\n")

        f.write(f"\n\nCASE 3 — HMM DEGRADED CORRECT CNN PREDICTIONS\n")
        f.write("Cases where context hurt accuracy — honest reporting.\n")

        for ex in cases["hmm_degraded"][:5]:
            f.write(f"\nTrue label   : {ex['true_name']}\n")
            f.write(f"CNN predicted: {ex['cnn_name']} \n")
            f.write(f"HMM decoded  : {ex['hmm_name']} \n")
            f.write(f"Entropy      : {ex['entropy']:.4f} ({ex['ent_norm'] * 100:.1f}% of max)\n")
            f.write(f"Confidence   : {ex['confidence'] * 100:.1f}%\n")

        f.write(f"\n\nSUMMARY:\n")
        f.write(f"HMM corrected CNN errors  : {len(cases['hmm_corrected'])}\n")
        f.write(f"Both correct              : {len(cases['both_correct'])}\n")
        f.write(f"Both wrong                : {len(cases['both_wrong'])}\n")
        f.write(f"HMM degraded CNN          : {len(cases['hmm_degraded'])}\n")
        total = sum(len(v) for v in cases.values())
        f.write(f"Total word slots          : {total}\n")

    corrected_entropies = [ex["entropy"] for ex in cases["hmm_corrected"]]
    both_correct_ents   = [ex["entropy"] for ex in cases["both_correct"]]
    degraded_entropies  = [ex["entropy"] for ex in cases["hmm_degraded"]]
    both_wrong_ents     = [ex["entropy"] for ex in cases["both_wrong"]]

    def mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_ent_corrected  = mean(corrected_entropies)
    avg_ent_correct    = mean(both_correct_ents)
    avg_ent_degraded   = mean(degraded_entropies)
    avg_ent_both_wrong = mean(both_wrong_ents)

    correlation_section = f"""
QUANTITATIVE ENTROPY CORRELATION:

Does the HMM help MORE when CNN entropy is HIGH?

  Outcome                    | Count | Mean Entropy | % of Max
  ---------------------------|-------|--------------|----------
  HMM corrected CNN error    | {len(corrected_entropies):>5} | {avg_ent_corrected:.4f}       | {avg_ent_corrected/max_ent*100:.1f}%
  Both correct (CNN + HMM)   | {len(both_correct_ents):>5} | {avg_ent_correct:.4f}       | {avg_ent_correct/max_ent*100:.1f}%
  HMM degraded CNN           | {len(degraded_entropies):>5} | {avg_ent_degraded:.4f}       | {avg_ent_degraded/max_ent*100:.1f}%
  Both wrong (CNN + HMM)     | {len(both_wrong_ents):>5} | {avg_ent_both_wrong:.4f}       | {avg_ent_both_wrong/max_ent*100:.1f}%

  Corrected mean entropy  : {avg_ent_corrected:.4f} ({avg_ent_corrected/max_ent*100:.1f}% of max)
  Both-correct entropy    : {avg_ent_correct:.4f} ({avg_ent_correct/max_ent*100:.1f}% of max)
  Difference              : {avg_ent_corrected - avg_ent_correct:+.4f}

  {"CONFIRMED: HMM corrects higher-entropy predictions than it reinforces." if avg_ent_corrected > avg_ent_correct else "NOT confirmed: further investigation needed."}

"""

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(correlation_section)

    print(f"Qualitative analysis saved → {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()