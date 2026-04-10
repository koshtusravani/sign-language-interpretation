"""
comparison_runner.py — Unified frame-based vs context-aware comparison.
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
    clip_emission_confidence_weighted,
    viterbi_decode,
    greedy_decode,
    TEMPERATURE,
)

FRAME_PREDS_FILE = "results/frame_predictions.pt"
SEQUENCES_FILE   = "results/sequences.pt"
RESULTS_DIR      = "results"
REPORT_FILE      = os.path.join(RESULTS_DIR, "comparison_report.txt")
JSON_FILE        = os.path.join(RESULTS_DIR, "comparison_results.json")


def evaluate_isolated(samples):
    all_logits = torch.stack([s["logits"].mean(dim=0) for s in samples])
    all_labels = torch.tensor([s["label"] for s in samples])
    all_probs  = F.softmax(all_logits, dim=1)

    top1 = top_k_accuracy(all_probs, all_labels, k=1)
    top3 = top_k_accuracy(all_probs, all_labels, k=3)
    top5 = top_k_accuracy(all_probs, all_labels, k=5)

    avg_correct = vote_correct = conf_correct = hmm1_correct = 0
    per_frame_entropies  = []
    clip_level_entropies = []

    for s in samples:
        logits     = s["logits"]
        probs      = F.softmax(logits, dim=1)
        true_label = s["label"]

        per_frame_entropies.append(prediction_entropy(probs).mean().item())

        avg_prob = probs.mean(dim=0)
        avg_pred = avg_prob.argmax().item()
        avg_correct += int(avg_pred == true_label)

        clip_level_entropies.append(prediction_entropy(avg_prob).item())

        confidences = probs.max(dim=1).values
        weights     = confidences / confidences.sum()
        conf_prob   = (probs * weights.unsqueeze(1)).sum(dim=0)
        conf_pred   = conf_prob.argmax().item()
        conf_correct += int(conf_pred == true_label)

        frame_preds = probs.argmax(dim=1)
        vals, cnts  = torch.unique(frame_preds, return_counts=True)
        vote_pred   = vals[cnts.argmax()].item()
        vote_correct += int(vote_pred == true_label)

        log_probs = torch.log(probs.clamp(min=1e-12))
        scores    = log_probs.sum(dim=0)
        hmm1_pred = scores.argmax().item()
        hmm1_correct += int(hmm1_pred == true_label)

    n = len(samples)
    return {
        "top1":                 top1,
        "top3":                 top3,
        "top5":                 top5,
        "avg_acc":              avg_correct  / n,
        "conf_acc":             conf_correct / n,
        "vote_acc":             vote_correct / n,
        "hmm1_acc":             hmm1_correct / n,
        "mean_per_frame_ent":   sum(per_frame_entropies)  / n,
        "mean_clip_level_ent":  sum(clip_level_entropies) / n,
        "total":                n,
    }


def evaluate_sequences(sequences):
    num_classes   = sequences[0]["clip_logits"][0].shape[1]
    log_trans_est = estimate_transition_from_sequences(sequences, num_classes)
    log_trans_uni = build_uniform_transition(num_classes, self_loop=0.05)

    counters = {k: 0 for k in [
        "slot_greedy", "slot_hmm_est_avg", "slot_hmm_est_conf", "slot_hmm_uni",
        "seq_greedy",  "seq_hmm_est_avg",  "seq_hmm_est_conf",  "seq_hmm_uni",
    ]}
    total_slots = 0
    total_seqs  = len(sequences)
    ent_before  = []

    for seq in sequences:
        clip_logits = seq["clip_logits"]
        true_labels = seq["clip_labels"]

        log_em_avg  = [clip_emission_log_probs(lg, TEMPERATURE)           for lg in clip_logits]
        log_em_conf = [clip_emission_confidence_weighted(lg, TEMPERATURE) for lg in clip_logits]

        for le in log_em_avg:
            ent_before.append(prediction_entropy(le.exp()).item())

        g_path      = greedy_decode(log_em_avg)
        ea_path, _  = viterbi_decode(log_em_avg,  log_trans_est)
        ec_path, _  = viterbi_decode(log_em_conf, log_trans_est)
        u_path,  _  = viterbi_decode(log_em_avg,  log_trans_uni)

        for pg, pa, pc, pu, t in zip(g_path, ea_path, ec_path, u_path, true_labels):
            counters["slot_greedy"]       += int(pg == t)
            counters["slot_hmm_est_avg"]  += int(pa == t)
            counters["slot_hmm_est_conf"] += int(pc == t)
            counters["slot_hmm_uni"]      += int(pu == t)
            total_slots += 1

        counters["seq_greedy"]       += int(g_path  == true_labels)
        counters["seq_hmm_est_avg"]  += int(ea_path == true_labels)
        counters["seq_hmm_est_conf"] += int(ec_path == true_labels)
        counters["seq_hmm_uni"]      += int(u_path  == true_labels)

    return {
        "slot_greedy":        counters["slot_greedy"]       / total_slots,
        "slot_hmm_est_avg":   counters["slot_hmm_est_avg"]  / total_slots,
        "slot_hmm_est_conf":  counters["slot_hmm_est_conf"] / total_slots,
        "slot_hmm_uni":       counters["slot_hmm_uni"]      / total_slots,
        "seq_greedy":         counters["seq_greedy"]        / total_seqs,
        "seq_hmm_est_avg":    counters["seq_hmm_est_avg"]   / total_seqs,
        "seq_hmm_est_conf":   counters["seq_hmm_est_conf"]  / total_seqs,
        "seq_hmm_uni":        counters["seq_hmm_uni"]       / total_seqs,
        "entropy_before_ctx": sum(ent_before) / len(ent_before),
        "total_seqs":         total_seqs,
        "total_slots":        total_slots,
    }


def write_report(iso, seq, path):
    best_slot  = max(seq["slot_hmm_est_avg"], seq["slot_hmm_est_conf"])
    best_seq   = max(seq["seq_hmm_est_avg"],  seq["seq_hmm_est_conf"])
    slot_delta = best_slot - seq["slot_greedy"]
    seq_delta  = best_seq  - seq["seq_greedy"]

    lines = [
        "=" * 65,
        "SIGN LANGUAGE RECOGNITION — COMPARISON REPORT",
        "=" * 65,
        "",
        "PART 1 — ISOLATED WORD RECOGNITION",
        "-" * 65,
        f"  Samples evaluated              : {iso['total']}",
        f"  Top-1 Accuracy (avg pool)      : {iso['top1']  * 100:.2f}%",
        f"  Top-3 Accuracy (avg pool)      : {iso['top3']  * 100:.2f}%",
        f"  Top-5 Accuracy (avg pool)      : {iso['top5']  * 100:.2f}%",
        "",
        "  Per-method comparison:",
        f"    Frame Average                : {iso['avg_acc']  * 100:.2f}%",
        f"    Confidence-Weighted Average  : {iso['conf_acc'] * 100:.2f}%",
        f"    Majority Vote                : {iso['vote_acc'] * 100:.2f}%",
        f"    Single-word HMM              : {iso['hmm1_acc'] * 100:.2f}%",
        "",
        "  Uncertainty Analysis (Shannon entropy):",
        f"    Mean per-frame entropy       : {iso['mean_per_frame_ent']:.4f}",
        f"    Mean clip-level entropy      : {iso['mean_clip_level_ent']:.4f}",
        "    Note: per-frame entropy measures individual frame uncertainty.",
        "    Clip-level entropy measures the aggregated prediction uncertainty.",
        "",
        "PART 2 — SEQUENCE RECOGNITION (multi-word HMM)",
        "-" * 65,
        f"  Sequences evaluated            : {seq['total_seqs']}",
        f"  Total word slots               : {seq['total_slots']}",
        f"  Temperature scaling            : {TEMPERATURE}",
        "",
        "  Slot-level accuracy (each word position):",
        f"    Greedy / no context          : {seq['slot_greedy']       * 100:.2f}%",
        f"    HMM (est trans + avg emit)   : {seq['slot_hmm_est_avg']  * 100:.2f}%",
        f"    HMM (est trans + conf emit)  : {seq['slot_hmm_est_conf'] * 100:.2f}%",
        f"    HMM (uniform transitions)    : {seq['slot_hmm_uni']      * 100:.2f}%",
        "",
        "  Sequence-level accuracy (all words correct):",
        f"    Greedy / no context          : {seq['seq_greedy']        * 100:.2f}%",
        f"    HMM (est trans + avg emit)   : {seq['seq_hmm_est_avg']   * 100:.2f}%",
        f"    HMM (est trans + conf emit)  : {seq['seq_hmm_est_conf']  * 100:.2f}%",
        f"    HMM (uniform transitions)    : {seq['seq_hmm_uni']       * 100:.2f}%",
        "",
        f"  Mean pre-decoding entropy      : {seq['entropy_before_ctx']:.4f}",
        "",
        "CONTEXTUAL BENEFIT SUMMARY",
        "-" * 65,
        f"  Best HMM vs Greedy — slot Δ   : {slot_delta * 100:+.2f}%",
        f"  Best HMM vs Greedy — seq  Δ   : {seq_delta  * 100:+.2f}%",
        "",
    ]

    if slot_delta > 0:
        lines.append("  ✓ Contextual modelling improved slot-level accuracy.")
    else:
        lines.append("  ✗ No slot-level improvement from context.")

    if seq_delta > 0:
        lines.append("  ✓ Contextual modelling improved sequence-level accuracy.")

    lines.append("=" * 65)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    print(f"\nReport saved → {path}")


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
            "Top-1 (avg pool)":        iso["top1"],
            "Top-3 (avg pool)":        iso["top3"],
            "Top-5 (avg pool)":        iso["top5"],
            "Frame Avg Acc":           iso["avg_acc"],
            "Conf-Weighted Acc":       iso["conf_acc"],
            "Majority Vote Acc":       iso["vote_acc"],
            "Single-word HMM Acc":     iso["hmm1_acc"],
            "Mean Per-Frame Entropy":  iso["mean_per_frame_ent"],
            "Mean Clip-Level Entropy": iso["mean_clip_level_ent"],
        },
        title="Isolated Word Metrics",
    )
    print()
    print_results_table(
        {
            "Slot – Greedy":           seq["slot_greedy"],
            "Slot – HMM est+avg":      seq["slot_hmm_est_avg"],
            "Slot – HMM est+conf":     seq["slot_hmm_est_conf"],
            "Slot – HMM uni":          seq["slot_hmm_uni"],
            "Seq  – Greedy":           seq["seq_greedy"],
            "Seq  – HMM est+avg":      seq["seq_hmm_est_avg"],
            "Seq  – HMM est+conf":     seq["seq_hmm_est_conf"],
            "Seq  – HMM uni":          seq["seq_hmm_uni"],
            "Mean Pre-Decode Entropy": seq["entropy_before_ctx"],
        },
        title="Sequence Metrics",
    )

    write_report(iso, seq, REPORT_FILE)
    save_json({"isolated": iso, "sequence": seq}, JSON_FILE)


if __name__ == "__main__":
    main()