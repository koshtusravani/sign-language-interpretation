import torch
import torch.nn.functional as F


def average_probabilities(frame_probs):
    """
    frame_probs: Tensor of shape (T, num_classes)
    Returns: Tensor of shape (num_classes,)
    """
    return torch.mean(frame_probs, dim=0)


def max_probabilities(frame_probs):
    """
    frame_probs: Tensor of shape (T, num_classes)
    Returns: Tensor of shape (num_classes,)
    """
    return torch.max(frame_probs, dim=0).values


def majority_vote(frame_preds):
    """
    frame_preds: Tensor of shape (T,)
    Returns: single predicted class index
    """
    values, counts = torch.unique(frame_preds, return_counts=True)
    return values[torch.argmax(counts)]


def aggregate_predictions(frame_logits):
    """
    frame_logits: Tensor of shape (T, num_classes)

    Returns:
        dict with:
        - avg_pred
        - max_pred
        - vote_pred
        - avg_probs
        - max_probs
    """
    probs = F.softmax(frame_logits, dim=1)

    avg_probs = average_probabilities(probs)
    max_probs = max_probabilities(probs)

    avg_pred = torch.argmax(avg_probs)
    max_pred = torch.argmax(max_probs)

    frame_preds = torch.argmax(probs, dim=1)
    vote_pred = majority_vote(frame_preds)

    return {
        "avg_pred": avg_pred.item(),
        "max_pred": max_pred.item(),
        "vote_pred": vote_pred.item(),
        "avg_probs": avg_probs,
        "max_probs": max_probs
    }


if __name__ == "__main__":
    # small test
    dummy_logits = torch.randn(20, 10)
    result = aggregate_predictions(dummy_logits)

    print("Average Prediction:", result["avg_pred"])
    print("Max Prediction:", result["max_pred"])
    print("Majority Vote Prediction:", result["vote_pred"])