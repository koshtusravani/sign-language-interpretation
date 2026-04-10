"""
comparison_runner.py — Unified frame-based vs context-aware comparison.

Reads:   results/frame_predictions.pt   (from wlasl_evaluate.py)
         results/sequences.pt           (from sequence_builder.py)
Writes:  results/comparison_report.txt
         results/comparison_results.json
"""

import os
import torch
import torch.nn.functional as F

from utils import (
    top_k_accuracy,
    prediction_entropy,
    print_results_table,
    save_json,
)
from hmm_sequence import (
    build_uniform_transition,
    estimate_transition_from_sequences,
    clip_emission_log_probs,
    viterbi_decode,
    greedy_decode,
)

FRAME_PREDS_FILE = "results/frame_predictions.pt"
SEQUENCES_FILE   = "results/sequences.pt"
RESULTS_DIR      = "results"
REPORT_FILE      = os.path.join(RESULTS_DIR, "comparison_report.txt")
JSON_FILE        = os.path.join(RESULTS_DIR, "comparison_results.json")


# ---------------------------------------------------------------------------
# Part 1 — Isolated word recognition
# ---------------------------------------------------------------------------

def evaluate_isolated(samples):
    num_classes = samples[0]["logits"].shape[1]

    all_logits = torch.stack([s["logits"].mean(dim=0) for s in samples])
    all_labels = torch.tensor([s["label"] for s in samples])
    all_probs  = F.softmax(all_logits, dim=1)

    top1 = top_k_accuracy(all_probs, all_labels, k=1)
    top3 = top_k_accuracy(all_probs, all_labels, k=3)
    top5 = top_k_accuracy(all_probs, all_labels, k=5)

    avg_correct = vote_correct = hmm1_correct = 0
    clip_ent_before = []
    clip_ent_after  = []

    for s in samples:
        logits     = s["logits"]
        probs      = F.softmax(logits, dim=1)
        true_label = s["label"]

        clip_ent_before.append(prediction_entropy(probs).mean().item())

        avg_prob  = probs.mean(dim=0)
        avg_pred  = avg_prob.argmax().item()
        avg_correct += int(avg_pred == true_label)

        frame_preds = probs.argmax(dim=1)
        vals, cnts  = torch.unique(frame_preds, return_counts=True)
        vote_pred   = vals[cnts.argmax()].item()
        vote_correct += int(vote_pred == true_label)

        log_probs = torch.log(probs.clamp(min=1e-12))
        scores    = log_probs.sum(dim=0)
        hmm1_pred = scores.argmax().item()
        hmm1_correct += int(hmm1_pred == true_label)

        clip_ent_after.append(prediction_entropy(avg_prob).item())

    n = len(samples)
    return {
        "top1":           top1,
        "top3":           top3,
        "top5":           top5,
        "avg_acc":        avg_correct  / n,
        "vote_acc":       vote_correct / n,
        "hmm1_acc":       hmm1_correct / n,
        "entropy_before": sum(clip_ent_before) / n,
        "entropy_after":  sum(clip_ent_after)  / n,
        "total":          n,
    }


# ---------------------------------------------------------------------------
# Part 2 — Sequence recognition
# ---------------------------------------------------------------------------

def evaluate_sequences(sequences):
    num_classes   = sequences[0]["clip_logits"][0].shape[1]
    log_trans_est = estimate_transition_from_sequences(sequences, num_classes)
    log_trans_uni = build_uniform_transition(num_classes, self_loop=0.05)

    slot_greedy = slot_hmm_est = slot_hmm_uni = 0
    seq_greedy  = seq_hmm_est  = seq_hmm_uni  = 0
    total_slots = 0
    total_seqs  = len(sequences)
    ent_before  = []

    for seq in sequences:
        clip_logits = seq["clip_logits"]
        true_labels = seq["clip_labels"]
        log_em      = [clip_emission_log_probs(lg) for lg in clip_logits]

        for le in log_em:
            ent_before.append(prediction_entropy(le.exp()).item())

        g_path          = greedy_decode(log_em)
        e_path, _       = viterbi_decode(log_em, log_trans_est)
        u_path, _       = viterbi_decode(log_em, log_trans_uni)

        for pg, pe, pu, t in zip(g_path, e_path, u_path, true_labels):
            slot_greedy  += int(pg == t)
            slot_hmm_est += int(pe == t)
            slot_hmm_uni += int(pu == t)
            total_slots  += 1

        seq_greedy  += int(g_path == true_labels)
        seq_hmm_est += int(e_path == true_labels)
        seq_hmm_uni += int(u_path == true_labels)

    return {
        "slot_greedy":        slot_greedy  / total_slots,
        "slot_hmm_est":       slot_hmm_est / total_slots,
        "slot_hmm_uni":       slot_hmm_uni / total_slots,
        "seq_greedy":         seq_greedy   / total_seqs,
        "seq_hmm_est":        seq_hmm_est  / total_seqs,
        "seq_hmm_uni":        seq_hmm_uni  / total_seqs,
        "entropy_before_ctx": sum(ent_before) / len(ent_before),
        "total_seqs":         total_seqs,
        "total_slots":        total_slots,
    }


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(iso, seq, path):
    slot_delta = seq["slot_hmm_est"] - seq["slot_greedy"]
    seq_delta  = seq["seq_hmm_est"]  - seq["seq_greedy"]

    lines = [
        "=" * 65,
        "SIGN LANGUAGE RECOGNITION — COMPARISON REPORT",
        "=" * 65,
        "",
        "PART 1 — ISOLATED WORD RECOGNITION",
        "-" * 65,
        f"  Samples evaluated            : {iso['total']}",
        f"  Top-1 Accuracy (avg pool)    : {iso['top1']  * 100:.2f}%",
        f"  Top-3 Accuracy (avg pool)    : {iso['top3']  * 100:.2f}%",
        f"  Top-5 Accuracy (avg pool)    : {iso['top5']  * 100:.2f}%",
        "",
        "  Per-method comparison:",
        f"    Frame Average              : {iso['avg_acc']  * 100:.2f}%",
        f"    Majority Vote              : {iso['vote_acc'] * 100:.2f}%",
        f"    Single-word HMM            : {iso['hmm1_acc'] * 100:.2f}%",
        "",
        "  Uncertainty (Shannon entropy):",
        f"    Before aggregation (mean)  : {iso['entropy_before']:.4f}",
        f"    After  aggregation (mean)  : {iso['entropy_after']:.4f}",
        f"    Reduction                  : {iso['entropy_before'] - iso['entropy_after']:.4f}",
        "",
        "PART 2 — SEQUENCE RECOGNITION (multi-word HMM)",
        "-" * 65,
        f"  Sequences evaluated          : {seq['total_seqs']}",
        f"  Total word slots             : {seq['total_slots']}",
        "",
        "  Slot-level accuracy:",
        f"    Greedy / no context        : {seq['slot_greedy']  * 100:.2f}%",
        f"    HMM (estimated transitions): {seq['slot_hmm_est'] * 100:.2f}%",
        f"    HMM (uniform  transitions) : {seq['slot_hmm_uni'] * 100:.2f}%",
        "",
        "  Sequence-level accuracy:",
        f"    Greedy / no context        : {seq['seq_greedy']   * 100:.2f}%",
        f"    HMM (estimated transitions): {seq['seq_hmm_est']  * 100:.2f}%",
        f"    HMM (uniform  transitions) : {seq['seq_hmm_uni']  * 100:.2f}%",
        "",
        f"  Mean pre-decoding entropy    : {seq['entropy_before_ctx']:.4f}",
        "",
        "CONTEXTUAL BENEFIT SUMMARY",
        "-" * 65,
        f"  HMM vs Greedy — slot-level  Δ : {slot_delta * 100:+.2f}%",
        f"  HMM vs Greedy — seq-level   Δ : {seq_delta  * 100:+.2f}%",
        "",
    ]

    if slot_delta > 0:
        lines.append("  ✓ Contextual modelling improved slot-level accuracy.")
    else:
        lines.append("  ✗ Contextual modelling did not improve slot-level accuracy.")
        lines.append("    (May indicate high CNN confidence — itself a finding.)")

    lines.append("=" * 65)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nReport saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    for fpath in [FRAME_PREDS_FILE, SEQUENCES_FILE]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing {fpath}.\n"
                "Run wlasl_evaluate.py then sequence_builder.py first."
            )

    print("Loading frame predictions …")
    samples = torch.load(FRAME_PREDS_FILE)

    print("Loading sequences …")
    sequences = torch.load(SEQUENCES_FILE)

    print("\n── Isolated word evaluation ──────────────────────────────")
    iso = evaluate_isolated(samples)

    print("\n── Sequence evaluation ───────────────────────────────────")
    seq = evaluate_sequences(sequences)

    print()
    print_results_table(
        {
            "Top-1 (avg pool)":    iso["top1"],
            "Top-3 (avg pool)":    iso["top3"],
            "Top-5 (avg pool)":    iso["top5"],
            "Frame Avg Acc":       iso["avg_acc"],
            "Majority Vote Acc":   iso["vote_acc"],
            "Single-word HMM Acc": iso["hmm1_acc"],
            "Entropy Before Agg":  iso["entropy_before"],
            "Entropy After Agg":   iso["entropy_after"],
        },
        title="Isolated Word Metrics",
    )
    print()
    print_results_table(
        {
            "Slot Acc – Greedy":        seq["slot_greedy"],
            "Slot Acc – HMM (est)":     seq["slot_hmm_est"],
            "Slot Acc – HMM (uni)":     seq["slot_hmm_uni"],
            "Seq  Acc – Greedy":        seq["seq_greedy"],
            "Seq  Acc – HMM (est)":     seq["seq_hmm_est"],
            "Seq  Acc – HMM (uni)":     seq["seq_hmm_uni"],
            "Mean Pre-Decode Entropy":  seq["entropy_before_ctx"],
        },
        title="Sequence Metrics",
    )

    write_report(iso, seq, REPORT_FILE)
    save_json({"isolated": iso, "sequence": seq}, JSON_FILE)


if __name__ == "__main__":
    main()