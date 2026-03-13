import os
import numpy as np

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQUENCE_LENGTH = 10
NOISE_STRENGTH = 0.15
SEED = 42

np.random.seed(SEED)


def add_noise_to_probabilities(probabilities, noise_strength=0.15):
    
    noisy_probs = probabilities.copy()

    noise = np.random.uniform(
        low=-noise_strength,
        high=noise_strength,
        size=noisy_probs.shape
    )

    noisy_probs = noisy_probs + noise
    noisy_probs = np.clip(noisy_probs, 1e-8, None)

    noisy_probs = noisy_probs / noisy_probs.sum(axis=1, keepdims=True)

    return noisy_probs


def build_sequences(sequence_length=10, noise_strength=0.15):
    probs_path = os.path.join(RESULTS_DIR, "cnn_probabilities.npy")
    labels_path = os.path.join(RESULTS_DIR, "cnn_true_labels.npy")

    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Missing file: {probs_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing file: {labels_path}")

    probabilities = np.load(probs_path)
    true_labels = np.load(labels_path)

    noisy_probabilities = add_noise_to_probabilities(
        probabilities,
        noise_strength=noise_strength
    )

    total_samples = len(true_labels)
    usable_samples = (total_samples // sequence_length) * sequence_length

    probabilities = probabilities[:usable_samples]
    noisy_probabilities = noisy_probabilities[:usable_samples]
    true_labels = true_labels[:usable_samples]

    num_sequences = usable_samples // sequence_length
    num_classes = probabilities.shape[1]

    probabilities_seq = probabilities.reshape(num_sequences, sequence_length, num_classes)
    noisy_probabilities_seq = noisy_probabilities.reshape(num_sequences, sequence_length, num_classes)
    true_labels_seq = true_labels.reshape(num_sequences, sequence_length)

    np.save(
        os.path.join(RESULTS_DIR, "sequence_probabilities.npy"),
        probabilities_seq
    )
    np.save(
        os.path.join(RESULTS_DIR, "sequence_noisy_probabilities.npy"),
        noisy_probabilities_seq
    )
    np.save(
        os.path.join(RESULTS_DIR, "sequence_true_labels.npy"),
        true_labels_seq
    )

    with open(os.path.join(RESULTS_DIR, "sequence_info.txt"), "w") as f:
        f.write(f"Sequence length: {sequence_length}\n")
        f.write(f"Noise strength: {noise_strength}\n")
        f.write(f"Total usable samples: {usable_samples}\n")
        f.write(f"Number of sequences: {num_sequences}\n")
        f.write(f"Number of classes: {num_classes}\n")

    print("Saved:")
    print("- results/sequence_probabilities.npy")
    print("- results/sequence_noisy_probabilities.npy")
    print("- results/sequence_true_labels.npy")
    print("- results/sequence_info.txt")


if __name__ == "__main__":
    build_sequences(
        sequence_length=SEQUENCE_LENGTH,
        noise_strength=NOISE_STRENGTH
    )