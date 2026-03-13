import os
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class HMMDecoder:
    def __init__(self, num_classes: int, stay_prob: float = 0.8):
        self.num_classes = num_classes
        self.transition_matrix = self._build_transition_matrix(stay_prob)

    def _build_transition_matrix(self, stay_prob: float) -> np.ndarray:
        """
        Build a simple transition matrix.
        - High probability of staying in the same class
        - Remaining probability spread across other classes
        """
        transition = np.full(
            (self.num_classes, self.num_classes),
            (1.0 - stay_prob) / (self.num_classes - 1),
            dtype=np.float64
        )
        np.fill_diagonal(transition, stay_prob)
        return transition

    def viterbi_decode(self, emissions: np.ndarray) -> np.ndarray:
        """
        emissions: shape (T, N)
            T = number of samples / time steps
            N = number of classes
        returns:
            best path of class indices, shape (T,)
        """
        T, N = emissions.shape

        # avoid log(0)
        emissions = np.clip(emissions, 1e-12, 1.0)
        transitions = np.clip(self.transition_matrix, 1e-12, 1.0)

        log_emissions = np.log(emissions)
        log_transitions = np.log(transitions)

        dp = np.zeros((T, N), dtype=np.float64)
        backpointer = np.zeros((T, N), dtype=np.int32)

        # initial step
        dp[0] = log_emissions[0]

        # dynamic programming
        for t in range(1, T):
            for j in range(N):
                scores = dp[t - 1] + log_transitions[:, j]
                best_prev = np.argmax(scores)
                dp[t, j] = scores[best_prev] + log_emissions[t, j]
                backpointer[t, j] = best_prev

        # backtrack
        best_path = np.zeros(T, dtype=np.int32)
        best_path[-1] = np.argmax(dp[-1])

        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        return best_path


def main():
    probs_path = os.path.join(RESULTS_DIR, "cnn_probabilities.npy")
    labels_path = os.path.join(RESULTS_DIR, "cnn_true_labels.npy")

    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Missing file: {probs_path}")

    emissions = np.load(probs_path)
    num_classes = emissions.shape[1]

    decoder = HMMDecoder(num_classes=num_classes, stay_prob=0.8)
    decoded_sequence = decoder.viterbi_decode(emissions)

    np.save(os.path.join(RESULTS_DIR, "hmm_decoded_sequence.npy"), decoded_sequence)
    np.save(os.path.join(RESULTS_DIR, "hmm_transition_matrix.npy"), decoder.transition_matrix)

    # optional quick comparison if true labels exist
    if os.path.exists(labels_path):
        true_labels = np.load(labels_path)
        raw_preds = np.argmax(emissions, axis=1)

        raw_acc = (raw_preds == true_labels).mean()
        hmm_acc = (decoded_sequence == true_labels).mean()

        with open(os.path.join(RESULTS_DIR, "hmm_comparison.txt"), "w") as f:
            f.write(f"Raw CNN accuracy: {raw_acc * 100:.2f}%\n")
            f.write(f"HMM-decoded accuracy: {hmm_acc * 100:.2f}%\n")

        print(f"Raw CNN accuracy: {raw_acc * 100:.2f}%")
        print(f"HMM-decoded accuracy: {hmm_acc * 100:.2f}%")

    print("HMM decoded sequence saved to results/hmm_decoded_sequence.npy")


if __name__ == "__main__":
    main()