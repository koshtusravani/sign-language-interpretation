import os
import json
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------

def top_k_accuracy(logits_or_probs, labels, k=5):
    """
    Compute top-k accuracy.

    Args:
        logits_or_probs: Tensor of shape (N, num_classes)
        labels:          Tensor of shape (N,)
        k:               int

    Returns:
        float in [0, 1]
    """
    if logits_or_probs.shape[1] < k:
        k = logits_or_probs.shape[1]

    _, top_k_preds = torch.topk(logits_or_probs, k, dim=1)
    labels_expanded = labels.unsqueeze(1).expand_as(top_k_preds)
    correct = top_k_preds.eq(labels_expanded).any(dim=1).sum().item()
    return correct / len(labels)


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def prediction_entropy(probs):
    """
    Shannon entropy over a probability distribution.

    Args:
        probs: Tensor of shape (num_classes,) or (N, num_classes)

    Returns:
        Tensor — scalar entropy or (N,) entropies
    """
    probs = torch.clamp(probs, min=1e-12)
    return -torch.sum(probs * torch.log(probs), dim=-1)


def frame_entropy_stats(frame_probs):
    """
    Per-frame entropy statistics for a single video clip.

    Args:
        frame_probs: Tensor of shape (T, num_classes)

    Returns:
        dict with mean_entropy, max_entropy, min_entropy (all floats)
    """
    entropies = prediction_entropy(frame_probs)
    return {
        "mean_entropy": entropies.mean().item(),
        "max_entropy":  entropies.max().item(),
        "min_entropy":  entropies.min().item(),
    }


def confidence_score(probs):
    """
    Max probability as a simple confidence measure.

    Args:
        probs: Tensor of shape (num_classes,)

    Returns:
        float
    """
    return probs.max().item()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_results_table(results: dict, title: str = "Results"):
    width = 44
    print("┌" + "─" * width + "┐")
    print(f"│  {title:<{width - 2}}│")
    print("├" + "─" * width + "┤")
    for key, val in results.items():
        if isinstance(val, float):
            line = f"  {key:<28}: {val:>8.4f}"
        else:
            line = f"  {key:<28}: {str(val):>8}"
        print(f"│{line:<{width}}│")
    print("└" + "─" * width + "┘")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(data, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Saved → {path}")


def save_frame_predictions(sequences: list, path: str):
    """
    Persist a list of dicts:
        [{"logits": Tensor(T, C), "label": int}, ...]
    to disk so hmm_wordLevel.py and sequence_eval.py can load them.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(sequences, path)
    print(f"Saved frame predictions → {path}  ({len(sequences)} samples)")


if __name__ == "__main__":
    dummy_logits = torch.randn(8, 20)
    dummy_probs  = F.softmax(dummy_logits, dim=1)
    dummy_labels = torch.randint(0, 20, (8,))

    top1 = top_k_accuracy(dummy_probs, dummy_labels, k=1)
    top5 = top_k_accuracy(dummy_probs, dummy_labels, k=5)
    ent  = prediction_entropy(dummy_probs[0])

    print_results_table({
        "Top-1 Accuracy": top1,
        "Top-5 Accuracy": top5,
        "Entropy (sample 0)": ent.item(),
    }, title="utils.py smoke-test")