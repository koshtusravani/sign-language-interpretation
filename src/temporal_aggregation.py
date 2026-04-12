import torch
import torch.nn.functional as F


def average_probabilities(frame_probs):
    return torch.mean(frame_probs, dim=0)


def max_probabilities(frame_probs):
    return torch.max(frame_probs, dim=0).values


def majority_vote(frame_preds):
    values, counts = torch.unique(frame_preds, return_counts=True)
    return values[torch.argmax(counts)]


def aggregate_predictions(frame_logits):
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
    dummy_logits = torch.randn(20, 10)
    result = aggregate_predictions(dummy_logits)

    print("Average Prediction:", result["avg_pred"])
    print("Max Prediction:", result["max_pred"])
    print("Majority Vote Prediction:", result["vote_pred"])