import os
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class HMMDecoder:
    def __init__(self, num_classes: int, stay_prob: float = 0.8):
        self.num_classes = num_classes
        self.transition_matrix = self._build_transition_matrix(stay_prob)

    def _build_transition_matrix(self, stay_prob: float) -> np.ndarray:
        transition = np.full(
            (self.num_classes, self.num_classes),
            (1.0 - stay_prob) / (self.num_classes - 1),
            dtype=np.float64
        )
        np.fill_diagonal(transition, stay_prob)
        return transition

    def viterbi_decode(self, emissions: np.ndarray) -> np.ndarray:
        T, N = emissions.shape

        emissions = np.clip(emissions, 1e-12, 1.0)
        transitions = np.clip(self.transition_matrix, 1e-12, 1.0)

        log_emissions = np.log(emissions)
        log_transitions = np.log(transitions)

        dp = np.zeros((T, N), dtype=np.float64)
        backpointer = np.zeros((T, N), dtype=np.int32)

        dp[0] = log_emissions[0]

        for t in range(1, T):
            for j in range(N):
                scores = dp[t - 1] + log_transitions[:, j]
                best_prev = np.argmax(scores)
                dp[t, j] = scores[best_prev] + log_emissions[t, j]
                backpointer[t, j] = best_prev

        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(dp[-1])

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean()


def main():
    noisy_probs_path = os.path.join(RESULTS_DIR, "sequence_noisy_probabilities.npy")
    clean_probs_path = os.path.join(RESULTS_DIR, "sequence_probabilities.npy")
    labels_path = os.path.join(RESULTS_DIR, "sequence_true_labels.npy")

    if not os.path.exists(noisy_probs_path):
        raise FileNotFoundError(f"Missing file: {noisy_probs_path}")
    if not os.path.exists(clean_probs_path):
        raise FileNotFoundError(f"Missing file: {clean_probs_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    noisy_probs = np.load(noisy_probs_path)
    clean_probs = np.load(clean_probs_path)
    true_labels = np.load(labels_path)

    num_sequences, sequence_length, num_classes = noisy_probs.shape

    decoder = HMMDecoder(num_classes=num_classes, stay_prob=0.8)

    raw_clean_preds = np.argmax(clean_probs, axis=2)
    raw_noisy_preds = np.argmax(noisy_probs, axis=2)

    hmm_preds = np.zeros_like(raw_noisy_preds)

    for i in range(num_sequences):
        hmm_preds[i] = decoder.viterbi_decode(noisy_probs[i])

    clean_acc = compute_accuracy(raw_clean_preds, true_labels)
    noisy_acc = compute_accuracy(raw_noisy_preds, true_labels)
    hmm_acc = compute_accuracy(hmm_preds, true_labels)

    np.save(os.path.join(RESULTS_DIR, "hmm_sequence_predictions.npy"), hmm_preds)
    np.save(os.path.join(RESULTS_DIR, "hmm_transition_matrix.npy"), decoder.transition_matrix)

    with open(os.path.join(RESULTS_DIR, "hmm_sequence_comparison.txt"), "w") as f:
        f.write(f"Raw CNN accuracy (clean probabilities): {clean_acc * 100:.2f}%\n")
        f.write(f"Raw CNN accuracy (noisy probabilities): {noisy_acc * 100:.2f}%\n")
        f.write(f"HMM-decoded accuracy (noisy probabilities): {hmm_acc * 100:.2f}%\n")

    print("Saved:")
    print("- results/hmm_sequence_predictions.npy")
    print("- results/hmm_transition_matrix.npy")
    print("- results/hmm_sequence_comparison.txt")


if __name__ == "__main__":
    main()