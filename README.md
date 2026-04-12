# Uncertainty-Aware Contextual Recognition of Sign Language Sequences

Team: Hema Sravani Koshtu | Dheeraj Royal Galla
Course: CS 5100 — Foundations of Artificial Intelligence, Northeastern University

A sign language recognition system that combines a ResNet18 CNN with a Hidden Markov Model (HMM) for uncertainty-aware contextual sequence decoding on the WLASL dataset. The system quantifies prediction uncertainty using Shannon entropy and uses a Viterbi HMM decoder to resolve high-uncertainty CNN predictions using word transition context. A REST API exposes the full inference pipeline as a deployable web service.

---

## Project Structure

    sign_language_interpretation/
    |
    |-- src/
    |   |-- wlasl_organize.py          # Organise raw WLASL videos by word class
    |   |-- wlasl_preprocess.py        # Extract frames from videos using OpenCV
    |   |-- wlasl_dataloader.py        # PyTorch dataset with train/test split + augmentation
    |   |-- wlasl_train.py             # Train ResNet18 CNN classifier
    |   |-- wlasl_evaluate.py          # Evaluate CNN, save per-frame predictions
    |   |-- utils.py                   # Shared helpers: top-k accuracy, entropy, logging
    |   |-- temporal_aggregation.py    # Frame aggregation methods (avg, max, vote)
    |   |-- hmm_wordLevel.py           # Isolated word HMM evaluation
    |   |-- sequence_builder.py        # Build synthetic multi-word sequences
    |   |-- hmm_sequence.py            # Multi-word Viterbi HMM decoder
    |   |-- comparison_runner.py       # Unified frame-based vs HMM comparison
    |   |-- qualitative_analysis.py    # Entropy correlation + qualitative examples
    |   |-- ethics_statement.py        # Generate ethics statement
    |   |-- generate_figures.py        # Generate paper figures from results
    |   |-- live_demo_opencv.py        # Live webcam demo — frame-based CNN
    |   |-- live_demo_hmm.py           # Live webcam demo — CNN + HMM side by side
    |   |-- api.py                     # REST API — Flask inference server
    |   |-- test_api.py                # API end-to-end tests (standard + --verify mode)
    |
    |-- data/
    |   |-- wlasl/
    |       |-- raw_videos/            # Organised raw .mp4 files (git-ignored)
    |       |-- frames/                # Extracted JPEG frames (git-ignored)
    |
    |-- archive/                       # Raw WLASL dataset files (git-ignored)
    |   |-- WLASL_v0.3.json
    |   |-- nslt_100.json
    |   |-- videos/
    |
    |-- models/                        # Saved model weights (git-ignored)
    |   |-- wlasl_word_model.pth       # Final epoch model
    |   |-- wlasl_word_model_best.pth  # Best checkpoint (used for evaluation)
    |
    |-- results/                       
    |   |-- frame_predictions.pt
    |   |-- sequences.pt
    |   |-- comparison_report.txt
    |   |-- hmm_sequence_results.txt
    |   |-- qualitative_analysis.txt
    |   |-- uncertainty_summary.json
    |   |-- ethics_statement.txt
    |   |-- fig1_training_loss.png
    |   |-- fig2_accuracy_comparison.png
    |   |-- fig3_per_class.png
    |   |-- fig4_entropy_outcomes.png
    |
    |-- requirements.txt
    |-- README.md
    |-- .gitignore
    |-- FINAL PROGRESS REPORT.pdf
    |-- FIRST PROGRESS REPORT.pdf
    |-- ACADEMIC PAPER.pdf

---

## Setup

### 1. Install dependencies

    pip install torch torchvision opencv-python mediapipe scikit-learn pillow numpy flask matplotlib

Or install from the requirements file:

    pip install -r requirements.txt

Note: MediaPipe is only required for the live demo scripts. Flask is only required for the API. matplotlib is only required for figure generation. All other scripts run without them.

### 2. Download the WLASL dataset

Download and place the following in archive/

- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed?resource=download

- WLASL_v0.3.json — full metadata
- nslt_100.json — 100-word subset annotations
- videos/ — raw .mp4 video files

---

## Pipeline — Run in This Order

### CNN Branch (wlasl-cnn)

Step 1 — Organise raw videos by word class

    python src/wlasl_organize.py

Step 2 — Extract frames using OpenCV

    python src/wlasl_preprocess.py

Step 3 — Train ResNet18 classifier (40 epochs, approx 5 minutes on CPU)

    python src/wlasl_train.py

Step 4 — Evaluate on test set and save per-frame predictions

    python src/wlasl_evaluate.py

### Sequence Branch (wlasl-sequence)

Step 5 — Build synthetic multi-word sequences

    python src/sequence_builder.py

Step 6 — Run multi-word HMM Viterbi decoder

    python src/hmm_sequence.py

Step 7 — Unified comparison report (frame-based vs HMM)

    python src/comparison_runner.py

Step 8 — Qualitative uncertainty analysis and entropy correlation

    python src/qualitative_analysis.py

Step 9 — Generate ethics statement

    python src/ethics_statement.py

Step 10 — Generate figures for paper in steps 4 to 8

    python src/generate_figures.py

### Optional

Isolated word HMM analysis — run any time after Step 4

    python src/hmm_wordLevel.py

Live webcam demo — frame-based CNN

    python src/live_demo_opencv.py

Live webcam demo — CNN vs HMM side by side

    python src/live_demo_hmm.py

Note: Steps 5 through 10 require results/frame_predictions.pt produced by Step 4.
The live demos require a webcam and mediapipe installed.

---

## REST API

The inference pipeline is also available as a Flask REST API.

### Start the server

    python src/api.py

The server starts on http://127.0.0.1:5000. Requires the trained model at models/wlasl_word_model_best.pth (run Step 3 and 4 first).

### Endpoints

| Method | Endpoint                       | Description                                              |
|--------|--------------------------------|----------------------------------------------------------|
| GET    | /health                        | Liveness check — returns model status and class list     |
| POST   | /predict/clip                  | Single clip prediction from base64-encoded JPEG frames   |
| POST   | /predict/sequence              | Multi-clip Viterbi sequence decoding from JPEG frames    |
| POST   | /predict/logits                | Single clip prediction from raw logit tensor             |
| POST   | /predict/sequence/logits       | Sequence decoding from raw logit tensors                 |

### Example — health check

    curl http://127.0.0.1:5000/health

### Example — single clip prediction

    POST /predict/clip
    { "frames": ["<base64-jpeg>", "<base64-jpeg>", ...] }

    Response:
    {
        "prediction": "basketball",
        "confidence": 0.8231,
        "entropy": 0.4102,
        "top3": [
            {"label": "basketball", "probability": 0.8231},
            {"label": "play",       "probability": 0.1104},
            {"label": "man",        "probability": 0.0412}
        ]
    }

### Example — sequence prediction

    POST /predict/sequence
    { "clips": [ [<frames>], [<frames>], [<frames>] ] }

    Response:
    {
        "sequence": ["who", "man", "play"],
        "slots": [
            {"greedy": "who",  "hmm": "who",  "confidence": 0.71, "entropy": 0.52},
            {"greedy": "man",  "hmm": "man",  "confidence": 0.64, "entropy": 0.68},
            {"greedy": "who",  "hmm": "play", "confidence": 0.29, "entropy": 1.94}
        ],
        "mean_entropy": 1.05
    }

### Run API tests

Standard end-to-end test (synthetic frames, verifies HTTP round-trip):

    python src/test_api.py

Verification mode (injects stored logits, confirms API matches offline evaluation exactly):

    python src/test_api.py --verify

---

## Key Results

### CNN Baseline (isolated word recognition)

| Metric         | Value |
|----------------|-------|
| Top-1 Accuracy | 56%   |
| Top-3 Accuracy | 76%   |
| Top-5 Accuracy | 88%   |
| Best Train Loss | 11.89 (epoch 38) |
| Mean per-frame entropy | 1.29 nats |
| Mean clip-level entropy | 1.92 nats |

### HMM Sequence Decoding (300 sequences, 900 word slots)

| Method                          | Slot Accuracy | Sequence Accuracy | Slot     | Seq      |
|---------------------------------|---------------|-------------------|----------|----------|
| Greedy (no context)             | 56.22%        | 15.33%            | —        | —        |
| HMM — uniform transitions       | 56.22%        | 15.33%            | +0.00%   | +0.00%   |
| HMM — estimated + avg emission  | 60.67%        | 45.33%            | +4.45%   | +30.00%  |
| HMM — estimated + conf emission | 65.67%        | 49.67%            | +9.44%   | +34.33%  |

### Entropy Correlation

HMM-corrected slots have mean entropy 1.999 nats (86.8% of max) vs 1.852 nats (80.4%) for already-correct slots, confirming the HMM provides the greatest benefit precisely where the CNN is most uncertain.

---

## Vocabulary

10 ASL words from the WLASL nslt_100 split:

basketball, birthday, but, city, man, many, orange, play, shirt, who

Selected to include visually confusable sign pairs such as man/who and orange/many, deliberately creating high-uncertainty scenarios to test the HMM's disambiguation capability.

---

## Branch Structure

| Branch         | Owner               | Files                                                                                                                                                                               |
|----------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| wlasl-cnn      | Hema Sravani Koshtu | wlasl_organize.py, wlasl_preprocess.py, wlasl_dataloader.py, wlasl_train.py, wlasl_evaluate.py                                                                                     |
| wlasl-sequence | Dheeraj Royal Galla | utils.py, temporal_aggregation.py, hmm_wordLevel.py, sequence_builder.py, hmm_sequence.py, comparison_runner.py, qualitative_analysis.py, ethics_statement.py, generate_figures.py, live_demo_opencv.py, live_demo_hmm.py, api.py, test_api.py |

---

## Citation

O. Koller, S. Zargaran, H. Ney, and R. Bowden,
"Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition,"
BMVC, 2016.

D. Li, C. Rodriguez, X. Yu, and H. Li,
"Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison,"
WACV, 2020.