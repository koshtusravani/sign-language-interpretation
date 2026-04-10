import os
import torch
import torch.nn.functional as F

from utils import (
    top_k_accuracy,
    prediction_entropy,
    frame_entropy_stats,
    print_results_table,
    save_json,
)

INPUT_FILE  = "results/frame_predictions.pt"
OUTPUT_FILE = "results/hmm_word_results.txt"
JSON_FILE   = "results/hmm_word_results.json"


def majority_vote(frame_preds):
    values, counts = torch.unique(frame_preds, return_counts=True)
    return values[torch.argmax(counts)]


def average_prediction(frame_probs):
    avg_probs = torch.mean(frame_probs, dim=0)
    return torch.argmax(avg_probs), avg_probs


def hmm_single_word_prediction(frame_probs, stay_prob=0.95):
    """
    1-state-per-word HMM over the clip.
    Returns predicted class index and the score tensor (used for top-k).
    """
    frame_probs = torch.clamp(frame_probs, min=1e-12)
    T, num_classes = frame_probs.shape
    log_stay = torch.log(torch.tensor(stay_prob, dtype=torch.float32))

    scores = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        emission_score   = torch.sum(torch.log(frame_probs[:, c]))
        transition_score = (T - 1) * log_stay
        scores[c] = emission_score + transition_score

    return torch.argmax(scores), scores


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Missing file: {INPUT_FILE}\n"
            "Run wlasl_evaluate.py first to generate per-frame predictions."
        )

    sequences = torch.load(INPUT_FILE)
    num_classes = sequences[0]["logits"].shape[1]

    total        = 0
    avg_correct  = 0
    vote_correct = 0
    hmm_correct  = 0

    all_avg_probs  = []
    all_hmm_scores = []
    all_labels     = []

    per_sample_entropy_before = []
    per_sample_entropy_after  = []

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Word-Level HMM / Aggregation Results\n")
        f.write("=" * 50 + "\n\n")

        for i, sample in enumerate(sequences):
            logits     = sample["logits"]
            true_label = sample["label"]

            frame_probs = F.softmax(logits, dim=1)
            frame_preds = torch.argmax(frame_probs, dim=1)

            avg_pred, avg_probs   = average_prediction(frame_probs)
            vote_pred             = majority_vote(frame_preds)
            hmm_pred, hmm_scores  = hmm_single_word_prediction(frame_probs, stay_prob=0.95)

            avg_pred   = avg_pred.item()
            vote_pred  = vote_pred.item()
            hmm_pred   = hmm_pred.item()

            total        += 1
            avg_correct  += int(avg_pred  == true_label)
            vote_correct += int(vote_pred == true_label)
            hmm_correct  += int(hmm_pred  == true_label)

            all_avg_probs.append(avg_probs.unsqueeze(0))
            all_hmm_scores.append(hmm_scores.unsqueeze(0))
            all_labels.append(true_label)

            stats = frame_entropy_stats(frame_probs)
            per_sample_entropy_before.append(stats["mean_entropy"])

            agg_entropy = prediction_entropy(avg_probs).item()
            per_sample_entropy_after.append(agg_entropy)

            if i < 20:
                f.write(f"Sample {i}\n")
                f.write(f"  True Label             : {true_label}\n")
                f.write(f"  Average Prediction     : {avg_pred}\n")
                f.write(f"  Majority Vote          : {vote_pred}\n")
                f.write(f"  HMM Prediction         : {hmm_pred}\n")
                f.write(f"  Mean Frame Entropy     : {stats['mean_entropy']:.4f}\n")
                f.write(f"  Aggregated Entropy     : {agg_entropy:.4f}\n")
                f.write("-" * 50 + "\n")

        all_avg_probs_tensor  = torch.cat(all_avg_probs,  dim=0)
        all_hmm_scores_tensor = torch.cat(all_hmm_scores, dim=0)
        label_tensor          = torch.tensor(all_labels)

        avg_top1 = top_k_accuracy(all_avg_probs_tensor,  label_tensor, k=1)
        avg_top3 = top_k_accuracy(all_avg_probs_tensor,  label_tensor, k=3)
        avg_top5 = top_k_accuracy(all_avg_probs_tensor,  label_tensor, k=5)
        hmm_top1 = top_k_accuracy(all_hmm_scores_tensor, label_tensor, k=1)
        hmm_top3 = top_k_accuracy(all_hmm_scores_tensor, label_tensor, k=3)
        hmm_top5 = top_k_accuracy(all_hmm_scores_tensor, label_tensor, k=5)

        mean_ent_before   = sum(per_sample_entropy_before) / len(per_sample_entropy_before)
        mean_ent_after    = sum(per_sample_entropy_after)  / len(per_sample_entropy_after)
        entropy_reduction = mean_ent_before - mean_ent_after

        f.write("\nOverall Accuracy\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Aggregation  Top-1: {avg_correct / total * 100:.2f}%\n")
        f.write(f"Average Aggregation  Top-3: {avg_top3 * 100:.2f}%\n")
        f.write(f"Average Aggregation  Top-5: {avg_top5 * 100:.2f}%\n")
        f.write(f"Majority Vote        Top-1: {vote_correct / total * 100:.2f}%\n")
        f.write(f"HMM                  Top-1: {hmm_correct / total * 100:.2f}%\n")
        f.write(f"HMM                  Top-3: {hmm_top3 * 100:.2f}%\n")
        f.write(f"HMM                  Top-5: {hmm_top5 * 100:.2f}%\n")
        f.write("\nUncertainty\n")
        f.write(f"Mean Frame Entropy (before): {mean_ent_before:.4f}\n")
        f.write(f"Mean Agg   Entropy (after) : {mean_ent_after:.4f}\n")
        f.write(f"Entropy Reduction          : {entropy_reduction:.4f}\n")

    print_results_table(
        {
            "Avg Pool  Top-1":     avg_correct / total,
            "Avg Pool  Top-3":     avg_top3,
            "Avg Pool  Top-5":     avg_top5,
            "Vote      Top-1":     vote_correct / total,
            "HMM(1-word) Top-1":  hmm_correct / total,
            "HMM(1-word) Top-3":  hmm_top3,
            "HMM(1-word) Top-5":  hmm_top5,
            "Entropy Before":     mean_ent_before,
            "Entropy After":      mean_ent_after,
            "Entropy Reduction":  entropy_reduction,
        },
        title="Word-Level HMM Results",
    )

    save_json(
        {
            "avg_pool":  {"top1": avg_correct / total, "top3": avg_top3, "top5": avg_top5},
            "vote":      {"top1": vote_correct / total},
            "hmm_1word": {"top1": hmm_correct  / total, "top3": hmm_top3, "top5": hmm_top5},
            "uncertainty": {
                "mean_entropy_before": mean_ent_before,
                "mean_entropy_after":  mean_ent_after,
                "entropy_reduction":   entropy_reduction,
            },
        },
        JSON_FILE,
    )

    print(f"\nResults saved → {OUTPUT_FILE}")
    print(f"JSON    saved → {JSON_FILE}")


if __name__ == "__main__":
    main()