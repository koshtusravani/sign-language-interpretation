import os
import torch
import torch.nn.functional as F

INPUT_FILE = "results/frame_predictions.pt"
OUTPUT_FILE = "results/hmm_word_results.txt"


def majority_vote(frame_preds):
    values, counts = torch.unique(frame_preds, return_counts=True)
    return values[torch.argmax(counts)]


def average_prediction(frame_probs):
    avg_probs = torch.mean(frame_probs, dim=0)
    return torch.argmax(avg_probs)


def hmm_single_word_prediction(frame_probs, stay_prob=0.95):
    """
    frame_probs: (T, num_classes)

    For isolated word recognition:
    - assume one word generates the full clip
    - score each class over the full frame sequence
    - choose the class with the highest total log-likelihood

    This is effectively a simple 1-state-per-word HMM over the clip.
    """
    frame_probs = torch.clamp(frame_probs, min=1e-12)

    T, num_classes = frame_probs.shape

    # Strong self-loop prior for staying in the same word state
    log_stay = torch.log(torch.tensor(stay_prob, dtype=torch.float32))

    scores = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        # Sum log probabilities for this class across all frames
        emission_score = torch.sum(torch.log(frame_probs[:, c]))

        # Add self-loop contributions for remaining frames
        transition_score = (T - 1) * log_stay

        scores[c] = emission_score + transition_score

    return torch.argmax(scores)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing file: {INPUT_FILE}")

    sequences = torch.load(INPUT_FILE)

    total = 0
    avg_correct = 0
    vote_correct = 0
    hmm_correct = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Word-Level Sequence Results\n")
        f.write("=" * 40 + "\n\n")

        for i, sample in enumerate(sequences):
            logits = sample["logits"]   # (T, num_classes)
            true_label = sample["label"]

            frame_probs = F.softmax(logits, dim=1)
            frame_preds = torch.argmax(frame_probs, dim=1)

            avg_pred = average_prediction(frame_probs).item()
            vote_pred = majority_vote(frame_preds).item()
            hmm_pred = hmm_single_word_prediction(frame_probs, stay_prob=0.95).item()

            total += 1
            avg_correct += int(avg_pred == true_label)
            vote_correct += int(vote_pred == true_label)
            hmm_correct += int(hmm_pred == true_label)

            if i < 20:
                f.write(f"Sample {i}\n")
                f.write(f"True Label: {true_label}\n")
                f.write(f"Average Prediction: {avg_pred}\n")
                f.write(f"Majority Vote Prediction: {vote_pred}\n")
                f.write(f"HMM Prediction: {hmm_pred}\n")
                f.write("-" * 40 + "\n")

        avg_acc = avg_correct / total * 100
        vote_acc = vote_correct / total * 100
        hmm_acc = hmm_correct / total * 100

        f.write("\nOverall Accuracy\n")
        f.write("=" * 40 + "\n")
        f.write(f"Average Aggregation Accuracy: {avg_acc:.2f}%\n")
        f.write(f"Majority Vote Accuracy: {vote_acc:.2f}%\n")
        f.write(f"HMM Accuracy: {hmm_acc:.2f}%\n")

    print(f"Saved HMM results to {OUTPUT_FILE}")
    print(f"Average Aggregation Accuracy: {avg_acc:.2f}%")
    print(f"Majority Vote Accuracy: {vote_acc:.2f}%")
    print(f"HMM Accuracy: {hmm_acc:.2f}%")


if __name__ == "__main__":
    main()