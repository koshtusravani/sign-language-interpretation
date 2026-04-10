# Uncertainty-Aware Contextual Recognition of Sign Language Sequences

Team: Hema Sravani Koshtu | Dheeraj Royal Galla

A sign language recognition system that combines a ResNet18 CNN with a Hidden Markov Model (HMM) for uncertainty-aware contextual sequence decoding on the WLASL dataset.

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
    |   |-- live_demo_opencv.py        # Live webcam demo — frame-based CNN
    |   |-- live_demo_hmm.py           # Live webcam demo — CNN + HMM side by side
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
    |-- results/                       # All evaluation outputs
    |   |-- frame_predictions.pt
    |   |-- sequences.pt
    |   |-- comparison_report.txt
    |   |-- hmm_sequence_results.txt
    |   |-- qualitative_analysis.txt
    |   |-- uncertainty_summary.json
    |   |-- ethics_statement.txt
    |
    |-- requirements.txt
    |-- README.md

---

## Setup

### 1. Install dependencies

    pip install torch torchvision opencv-python mediapipe scikit-learn pillow numpy

Or install from the requirements file:

    pip install -r requirements.txt

Note: MediaPipe is only required for the live demo scripts. All other scripts run without it.

### 2. Download the WLASL dataset

Download from https://github.com/dxli94/WLASL and place the following in archive/

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

### Optional

Isolated word HMM analysis — run any time after Step 4

    python src/hmm_wordLevel.py

Live webcam demo — frame-based CNN

    python src/live_demo_opencv.py

Live webcam demo — CNN vs HMM side by side

    python src/live_demo_hmm.py

Note: Steps 5 through 9 require results/frame_predictions.pt produced by Step 4.
The live demos require a webcam and mediapipe installed.

---

## Key Results

| Method                              | Slot Accuracy | Sequence Accuracy |
|-------------------------------------|---------------|-------------------|
| Greedy (no context)                 | 53.67%        | 10.00%            |
| HMM — uniform transitions           | 54.67%        | 10.00%            |
| HMM — estimated + avg emission      | 64.00%        | 50.67%            |
| HMM — estimated + conf emission     | 68.78%        | 52.67%            |

CNN baseline: 56% top-1, 80% top-3, 96% top-5 accuracy on the 10-class test set.

Entropy correlation: HMM-corrected clips have mean entropy 2.001 nats (86.9% of max)
vs 1.822 nats (79.1%) for already-correct clips, confirming the HMM specifically
benefits high-uncertainty predictions.

---

## Vocabulary

10 ASL words from the WLASL nslt_100 split:

basketball, birthday, but, city, man, many, orange, play, shirt, who

Selected to include visually confusable sign pairs such as man/who and orange/many,
deliberately creating high-uncertainty scenarios to test the HMM's disambiguation capability.

---

## Branch Structure

| Branch         | Owner                  | Files |
|----------------|------------------------|-------|
| wlasl-cnn      | Hema Sravani Koshtu    | wlasl_organize.py, wlasl_preprocess.py, wlasl_dataloader.py, wlasl_train.py, wlasl_evaluate.py |
| wlasl-sequence | Dheeraj Royal Galla    | utils.py, temporal_aggregation.py, hmm_wordLevel.py, sequence_builder.py, hmm_sequence.py, comparison_runner.py, qualitative_analysis.py, ethics_statement.py, live_demo_opencv.py, live_demo_hmm.py |

---

## Citation

O. Koller, S. Zargaran, H. Ney, and R. Bowden,
"Deep Sign: Hybrid CNN-HMM for Continuous Sign Language Recognition,"
BMVC, 2016.

D. Li, C. Rodriguez, X. Yu, and H. Li,
"Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison,"
WACV, 2020.