import os
import numpy as np
import csv

RESULTS_DIR = "results"
METADATA_DIR = "data/metadata"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_class_names():
    class_file = os.path.join(METADATA_DIR, "class_names.txt")
    with open(class_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    return class_names


def compare_predictions():
    class_names = load_class_names()

    probs = np.load(os.path.join(RESULTS_DIR, "cnn_probabilities.npy"))
    true_labels = np.load(os.path.join(RESULTS_DIR, "cnn_true_labels.npy"))
    hmm_preds = np.load(os.path.join(RESULTS_DIR, "hmm_decoded_sequence.npy"))

    raw_preds = np.argmax(probs, axis=1)

    csv_path = os.path.join(RESULTS_DIR, "prediction_comparison.csv")
    txt_path = os.path.join(RESULTS_DIR, "sample_prediction_comparison.txt")

    # Save full comparison as CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "raw_cnn_prediction", "hmm_prediction"])

        for i in range(len(true_labels)):
            writer.writerow([
                i,
                class_names[true_labels[i]],
                class_names[raw_preds[i]],
                class_names[hmm_preds[i]]
            ])

    # Save first 50 examples as readable text
    with open(txt_path, "w") as f:
        f.write("Sample Prediction Comparison\n")
        f.write("=" * 50 + "\n\n")

        for i in range(min(50, len(true_labels))):
            f.write(
                f"Sample {i}: "
                f"True={class_names[true_labels[i]]}, "
                f"Raw CNN={class_names[raw_preds[i]]}, "
                f"HMM={class_names[hmm_preds[i]]}\n"
            )

    print("Saved:")
    print(f"- {csv_path}")
    print(f"- {txt_path}")


if __name__ == "__main__":
    compare_predictions()