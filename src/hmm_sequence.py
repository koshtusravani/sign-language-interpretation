"""
hmm_sequence.py — Multi-word HMM sequence decoder with Viterbi.

  - One state per vocabulary word
  - Emission  = frame-averaged softmax from the CNN
  - Transition matrix estimated from sequence data (or uniform fallback)
  - Viterbi decoding finds the most likely word sequence

Reads:   results/sequences.pt            (from sequence_builder.py)
Writes:  results/hmm_sequence_results.txt
         results/hmm_sequence_accuracy.json
"""

import os
import math
import torch
import torch.nn.functional as F

from utils import print_results_table, save_json, prediction_entropy

SEQUENCES_FILE = "results/sequences.pt"
RESULTS_DIR    = "results"
OUTPUT_TXT     = os.path.join(RESULTS_DIR, "hmm_sequence_results.txt")
OUTPUT_JSON    = os.path.join(RESULTS_DIR, "hmm_sequence_accuracy.json")


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def build_uniform_transition(num_classes: int, self_loop: float = 0.1):
    off_diag_prob = (1.0 - self_loop) / max(num_classes - 1, 1)
    trans = torch.full((num_classes, num_classes), off_diag_prob)
    trans.fill_diagonal_(self_loop)
    return torch.log(trans.clamp(min=1e-12))


def estimate_transition_from_sequences(sequences, num_classes: int):
    counts = torch.ones(num_classes, num_classes)   # Laplace smoothing
    for seq in sequences:
        labels = seq["clip_labels"]
        for i in range(len(labels) - 1):
            counts[labels[i], labels[i + 1]] += 1.0
    trans = counts / counts.sum(dim=1, keepdim=True)
    return torch.log(trans.clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def clip_emission_log_probs(clip_logits: torch.Tensor):
    """
    Summarise a clip into a single log-prob emission vector.
    Shape: (T, num_classes) → (num_classes,)
    """
    probs = F.softmax(clip_logits, dim=1)
    avg_probs = probs.mean(dim=0)
    return torch.log(avg_probs.clamp(min=1e-12))


# ---------------------------------------------------------------------------
# Viterbi decoder
# ---------------------------------------------------------------------------

def viterbi_decode(clip_log_emissions, log_trans, log_prior=None):
    """
    Args:
        clip_log_emissions: list of Tensors (num_classes,) — one per word slot
        log_trans:          Tensor (num_classes, num_classes)
        log_prior:          Tensor (num_classes,) or None

    Returns:
        best_path:  list of ints
        best_score: float
    """
    T           = len(clip_log_emissions)
    num_classes = clip_log_emissions[0].shape[0]

    if log_prior is None:
        log_prior = torch.full((num_classes,), -math.log(num_classes))

    viterbi = torch.zeros(T, num_classes)
    backptr = torch.zeros(T, num_classes, dtype=torch.long)

    viterbi[0] = log_prior + clip_log_emissions[0]

    for t in range(1, T):
        trans_scores = viterbi[t - 1].unsqueeze(1) + log_trans   # (C, C)
        best_prev_scores, best_prev_states = trans_scores.max(dim=0)
        viterbi[t] = best_prev_scores + clip_log_emissions[t]
        backptr[t] = best_prev_states

    best_last  = viterbi[-1].argmax().item()
    best_score = viterbi[-1, best_last].item()

    path = [best_last]
    for t in range(T - 1, 0, -1):
        best_last = backptr[t, best_last].item()
        path.append(best_last)

    path.reverse()
    return path, best_score


# ---------------------------------------------------------------------------
# Greedy baseline (no context)
# ---------------------------------------------------------------------------

def greedy_decode(clip_log_emissions):
    return [e.argmax().item() for e in clip_log_emissions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(SEQUENCES_FILE):
        raise FileNotFoundError(
            f"Missing {SEQUENCES_FILE}.\nRun sequence_builder.py first."
        )

    sequences   = torch.load(SEQUENCES_FILE)
    num_classes = sequences[0]["clip_logits"][0].shape[1]

    print(f"Loaded {len(sequences)} sequences  |  vocab size = {num_classes}")

    log_trans_estimated = estimate_transition_from_sequences(sequences, num_classes)
    log_trans_uniform   = build_uniform_transition(num_classes, self_loop=0.05)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    slot_greedy  = slot_hmm_est  = slot_hmm_uni  = 0
    seq_greedy   = seq_hmm_est   = seq_hmm_uni   = 0
    total_slots  = 0
    total_seqs   = len(sequences)
    entropy_before = []

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("HMM Sequence Decoding Results\n")
        f.write("=" * 60 + "\n\n")

        for i, seq in enumerate(sequences):
            clip_logits = seq["clip_logits"]
            true_labels = seq["clip_labels"]

            log_emissions = [clip_emission_log_probs(lg) for lg in clip_logits]

            for le in log_emissions:
                entropy_before.append(prediction_entropy(le.exp()).item())

            greedy_path             = greedy_decode(log_emissions)
            hmm_est_path, score_e   = viterbi_decode(log_emissions, log_trans_estimated)
            hmm_uni_path, score_u   = viterbi_decode(log_emissions, log_trans_uniform)

            for pg, pe, pu, true in zip(greedy_path, hmm_est_path, hmm_uni_path, true_labels):
                slot_greedy  += int(pg == true)
                slot_hmm_est += int(pe == true)
                slot_hmm_uni += int(pu == true)
                total_slots  += 1

            seq_greedy  += int(greedy_path  == true_labels)
            seq_hmm_est += int(hmm_est_path == true_labels)
            seq_hmm_uni += int(hmm_uni_path == true_labels)

            if i < 20:
                f.write(f"Sequence {i}\n")
                f.write(f"  True Labels          : {true_labels}\n")
                f.write(f"  Greedy (no context)  : {greedy_path}\n")
                f.write(f"  HMM (estimated trans): {hmm_est_path}  score={score_e:.3f}\n")
                f.write(f"  HMM (uniform  trans) : {hmm_uni_path}  score={score_u:.3f}\n")
                f.write("-" * 60 + "\n")

        sa_g  = slot_greedy  / total_slots
        sa_e  = slot_hmm_est / total_slots
        sa_u  = slot_hmm_uni / total_slots
        sq_g  = seq_greedy   / total_seqs
        sq_e  = seq_hmm_est  / total_seqs
        sq_u  = seq_hmm_uni  / total_seqs
        mean_ent = sum(entropy_before) / len(entropy_before)

        f.write("\n" + "=" * 60 + "\n")
        f.write("Slot-Level Accuracy\n")
        f.write(f"  Greedy              : {sa_g * 100:.2f}%\n")
        f.write(f"  HMM (estimated)     : {sa_e * 100:.2f}%\n")
        f.write(f"  HMM (uniform)       : {sa_u * 100:.2f}%\n")
        f.write("\nSequence-Level Accuracy\n")
        f.write(f"  Greedy              : {sq_g * 100:.2f}%\n")
        f.write(f"  HMM (estimated)     : {sq_e * 100:.2f}%\n")
        f.write(f"  HMM (uniform)       : {sq_u * 100:.2f}%\n")
        f.write(f"\nMean Pre-Decoding Entropy : {mean_ent:.4f}\n")

    print_results_table(
        {
            "Slot Acc – Greedy":       sa_g,
            "Slot Acc – HMM(est)":     sa_e,
            "Slot Acc – HMM(uni)":     sa_u,
            "Seq  Acc – Greedy":       sq_g,
            "Seq  Acc – HMM(est)":     sq_e,
            "Seq  Acc – HMM(uni)":     sq_u,
            "Mean Pre-Decode Entropy": mean_ent,
        },
        title="HMM Sequence Results",
    )

    save_json(
        {
            "slot_accuracy":  {"greedy": sa_g, "hmm_est": sa_e, "hmm_uni": sa_u},
            "sequence_accuracy": {"greedy": sq_g, "hmm_est": sq_e, "hmm_uni": sq_u},
            "mean_pre_decoding_entropy": mean_ent,
            "total_sequences": total_seqs,
            "total_slots":     total_slots,
        },
        OUTPUT_JSON,
    )

    print(f"\nDetailed results → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()