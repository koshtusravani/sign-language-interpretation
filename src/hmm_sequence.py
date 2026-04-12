import os
import math
import torch
import torch.nn.functional as F

from utils import print_results_table, save_json, prediction_entropy

SEQUENCES_FILE = "results/sequences.pt"
RESULTS_DIR    = "results"
OUTPUT_TXT     = os.path.join(RESULTS_DIR, "hmm_sequence_results.txt")
OUTPUT_JSON    = os.path.join(RESULTS_DIR, "hmm_sequence_accuracy.json")

TEMPERATURE = 0.5


def build_uniform_transition(num_classes: int, self_loop: float = 0.1):
    off_diag_prob = (1.0 - self_loop) / max(num_classes - 1, 1)
    trans = torch.full((num_classes, num_classes), off_diag_prob)
    trans.fill_diagonal_(self_loop)
    return torch.log(trans.clamp(min=1e-12))


def estimate_transition_from_sequences(sequences, num_classes: int):
    counts = torch.ones(num_classes, num_classes)
    for seq in sequences:
        labels = seq["clip_labels"]
        for i in range(len(labels) - 1):
            counts[labels[i], labels[i + 1]] += 1.0
    trans = counts / counts.sum(dim=1, keepdim=True)
    return torch.log(trans.clamp(min=1e-12))


def clip_emission_log_probs(clip_logits: torch.Tensor, temperature: float = 1.0):
    scaled_logits = clip_logits / temperature
    probs         = F.softmax(scaled_logits, dim=1)
    avg_probs     = probs.mean(dim=0)
    return torch.log(avg_probs.clamp(min=1e-12))


def clip_emission_confidence_weighted(clip_logits: torch.Tensor, temperature: float = 1.0):
    scaled_logits  = clip_logits / temperature
    probs          = F.softmax(scaled_logits, dim=1)
    confidences    = probs.max(dim=1).values
    weights        = confidences / confidences.sum()
    weighted_probs = (probs * weights.unsqueeze(1)).sum(dim=0)
    return torch.log(weighted_probs.clamp(min=1e-12))


def viterbi_decode(clip_log_emissions, log_trans, log_prior=None):
    T           = len(clip_log_emissions)
    num_classes = clip_log_emissions[0].shape[0]

    if log_prior is None:
        log_prior = torch.full((num_classes,), -math.log(num_classes))

    viterbi = torch.zeros(T, num_classes)
    backptr = torch.zeros(T, num_classes, dtype=torch.long)

    viterbi[0] = log_prior + clip_log_emissions[0]

    for t in range(1, T):
        trans_scores = viterbi[t - 1].unsqueeze(1) + log_trans
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


def greedy_decode(clip_log_emissions):
    return [e.argmax().item() for e in clip_log_emissions]


def main():
    if not os.path.exists(SEQUENCES_FILE):
        raise FileNotFoundError(
            f"Missing {SEQUENCES_FILE}.\nRun sequence_builder.py first."
        )

    sequences   = torch.load(SEQUENCES_FILE)
    num_classes = sequences[0]["clip_logits"][0].shape[1]
    print(f"Loaded {len(sequences)} sequences  |  vocab size = {num_classes}")
    print(f"Temperature scaling: {TEMPERATURE}")

    log_trans_est = estimate_transition_from_sequences(sequences, num_classes)
    log_trans_uni = build_uniform_transition(num_classes, self_loop=0.05)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    counters = {k: 0 for k in [
        "slot_greedy", "slot_hmm_est_avg", "slot_hmm_est_conf", "slot_hmm_uni",
        "seq_greedy",  "seq_hmm_est_avg",  "seq_hmm_est_conf",  "seq_hmm_uni",
    ]}
    total_slots    = 0
    total_seqs     = len(sequences)
    entropy_before = []

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("HMM Sequence Decoding Results\n")
        f.write(f"Temperature = {TEMPERATURE}\n")
        f.write("=" * 60 + "\n\n")

        for i, seq in enumerate(sequences):
            clip_logits = seq["clip_logits"]
            true_labels = seq["clip_labels"]

            log_em_avg  = [clip_emission_log_probs(lg, TEMPERATURE) for lg in clip_logits]
            log_em_conf = [clip_emission_confidence_weighted(lg, TEMPERATURE) for lg in clip_logits]

            for le in log_em_avg:
                entropy_before.append(prediction_entropy(le.exp()).item())

            g_path           = greedy_decode(log_em_avg)
            ea_path, _       = viterbi_decode(log_em_avg,  log_trans_est)
            ec_path, _       = viterbi_decode(log_em_conf, log_trans_est)
            u_path,  _       = viterbi_decode(log_em_avg,  log_trans_uni)

            for pg, pa, pc, pu, true in zip(g_path, ea_path, ec_path, u_path, true_labels):
                counters["slot_greedy"]       += int(pg == true)
                counters["slot_hmm_est_avg"]  += int(pa == true)
                counters["slot_hmm_est_conf"] += int(pc == true)
                counters["slot_hmm_uni"]      += int(pu == true)
                total_slots += 1

            counters["seq_greedy"]       += int(g_path  == true_labels)
            counters["seq_hmm_est_avg"]  += int(ea_path == true_labels)
            counters["seq_hmm_est_conf"] += int(ec_path == true_labels)
            counters["seq_hmm_uni"]      += int(u_path  == true_labels)

            if i < 20:
                f.write(f"Sequence {i}\n")
                f.write(f"  True         : {true_labels}\n")
                f.write(f"  Greedy       : {g_path}\n")
                f.write(f"  HMM est+avg  : {ea_path}\n")
                f.write(f"  HMM est+conf : {ec_path}\n")
                f.write(f"  HMM uni      : {u_path}\n")
                f.write("-" * 60 + "\n")

        sa_g  = counters["slot_greedy"]       / total_slots
        sa_ea = counters["slot_hmm_est_avg"]  / total_slots
        sa_ec = counters["slot_hmm_est_conf"] / total_slots
        sa_u  = counters["slot_hmm_uni"]      / total_slots
        sq_g  = counters["seq_greedy"]        / total_seqs
        sq_ea = counters["seq_hmm_est_avg"]   / total_seqs
        sq_ec = counters["seq_hmm_est_conf"]  / total_seqs
        sq_u  = counters["seq_hmm_uni"]       / total_seqs
        mean_ent = sum(entropy_before) / len(entropy_before)

        f.write("\n" + "=" * 60 + "\n")
        f.write("Slot-Level Accuracy\n")
        f.write(f"  Greedy                     : {sa_g  * 100:.2f}%\n")
        f.write(f"  HMM (est trans + avg emit) : {sa_ea * 100:.2f}%\n")
        f.write(f"  HMM (est trans + conf emit): {sa_ec * 100:.2f}%\n")
        f.write(f"  HMM (uniform trans)        : {sa_u  * 100:.2f}%\n")
        f.write("\nSequence-Level Accuracy\n")
        f.write(f"  Greedy                     : {sq_g  * 100:.2f}%\n")
        f.write(f"  HMM (est trans + avg emit) : {sq_ea * 100:.2f}%\n")
        f.write(f"  HMM (est trans + conf emit): {sq_ec * 100:.2f}%\n")
        f.write(f"  HMM (uniform trans)        : {sq_u  * 100:.2f}%\n")
        f.write(f"\nMean Pre-Decoding Entropy  : {mean_ent:.4f}\n")
        f.write(f"Temperature                : {TEMPERATURE}\n")

    print_results_table(
        {
            "Slot – Greedy":           sa_g,
            "Slot – HMM est+avg":      sa_ea,
            "Slot – HMM est+conf":     sa_ec,
            "Slot – HMM uni":          sa_u,
            "Seq  – Greedy":           sq_g,
            "Seq  – HMM est+avg":      sq_ea,
            "Seq  – HMM est+conf":     sq_ec,
            "Seq  – HMM uni":          sq_u,
            "Mean Pre-Decode Entropy": mean_ent,
            "Temperature":             TEMPERATURE,
        },
        title="HMM Sequence Results",
    )

    save_json(
        {
            "temperature": TEMPERATURE,
            "slot_accuracy": {
                "greedy":       sa_g,
                "hmm_est_avg":  sa_ea,
                "hmm_est_conf": sa_ec,
                "hmm_uni":      sa_u,
            },
            "sequence_accuracy": {
                "greedy":       sq_g,
                "hmm_est_avg":  sq_ea,
                "hmm_est_conf": sq_ec,
                "hmm_uni":      sq_u,
            },
            "mean_pre_decoding_entropy": mean_ent,
            "total_sequences": total_seqs,
            "total_slots":     total_slots,
        },
        OUTPUT_JSON,
    )

    print(f"\nDetailed results → {OUTPUT_TXT}")


if __name__ == "__main__":
    main()