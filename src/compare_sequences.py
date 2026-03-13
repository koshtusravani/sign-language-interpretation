import os
import numpy as np

RESULTS_DIR = "results"
METADATA_DIR = "data/metadata"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_class_names():
    with open(os.path.join(METADATA_DIR, "class_names.txt"), "r") as f:
        return [line.strip() for line in f if line.strip()]


def sequence_to_names(sequence, class_names):
    return [class_names[int(x)] for x in sequence]


def main():
    class_names = load_class_names()

    true_labels = np.load(os.path.join(RESULTS_DIR, "sequence_true_labels.npy"))
    noisy_probs = np.load(os.path.join(RESULTS_DIR, "sequence_noisy_probabilities.npy"))
    hmm_preds = np.load(os.path.join(RESULTS_DIR, "hmm_sequence_predictions.npy"))

    raw_noisy_preds = np.argmax(noisy_probs, axis=2)

    output_path = os.path.join(RESULTS_DIR, "sequence_examples.txt")

    with open(output_path, "w") as f:
        f.write("Sequence Comparison Examples\n")
        f.write("=" * 60 + "\n\n")

        shown = 0
        for i in range(len(true_labels)):
            true_seq = true_labels[i]
            raw_seq = raw_noisy_preds[i]
            hmm_seq = hmm_preds[i]

            # only show interesting examples where HMM differs from raw prediction
            if not np.array_equal(raw_seq, hmm_seq):
                f.write(f"Sequence {i}\n")
                f.write(f"True Labels : {' '.join(sequence_to_names(true_seq, class_names))}\n")
                f.write(f"Raw Noisy   : {' '.join(sequence_to_names(raw_seq, class_names))}\n")
                f.write(f"HMM Decoded : {' '.join(sequence_to_names(hmm_seq, class_names))}\n")
                f.write("-" * 60 + "\n")
                shown += 1

            if shown >= 20:
                break

        if shown == 0:
            f.write("No differing sequences found between raw noisy predictions and HMM predictions.\n")

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()