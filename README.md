# Uncertainty-Aware Contextual Recognition of Sign Language Sequences

## Project Overview

This project implements a sign language recognition system that combines deep learning (CNN) with probabilistic sequence modelling (HMM) to handle uncertainty in predictions.

The system focuses on word-level recognition using video data and demonstrates how probabilistic decoding can improve robustness over frame-level predictions.

## Objective

To design an AI-based system that:

* Recognizes sign language gestures from video input
* Produces probabilistic predictions using a CNN
* Applies HMM-based decoding to incorporate temporal consistency
* Handles uncertainty in predictions using probabilistic reasoning

## Methodology

### 1. CNN-based Visual Recognition
* Model: ResNet18 (PyTorch)
* Input: Extracted video frames
* Output: Frame-level probability distribution over sign classes

### 2. Temporal Processing
* Videos are preprocessed using segment-based frame extraction
* Only relevant sign segments are used (based on annotations)
* Reduces noise and improves model performance

### 3. Probabilistic Decoding (HMM)
* CNN outputs are treated as emission probabilities
* HMM assumes a single hidden word state per video
* Uses log-likelihood scoring across frames
* Produces final word prediction

### 4. Decoding Strategies
Different decoding strategies were evaluated:
* Direct CNN prediction (baseline)
* Average probability aggregation
* Majority voting
* HMM-based decoding

## Results
| Method              | Accuracy   |
| ------------------- | ---------- |
| CNN Baseline        | 98.06%     |
| Average Aggregation | 98.06%     |
| Majority Vote       | 94.17%     |
| CNN + HMM           | 98.06%     |

### Key Observations:

* Segment-based preprocessing significantly improved performance
* Probabilistic aggregation is more reliable than voting
* Properly formulated HMM matches CNN baseline performance
* Naive transition-based HMM can degrade performance

## Project Structure

sign-language-interpretation/
│
├── src/
│   ├── wlasl_organize.py
│   ├── wlasl_preprocess.py
│   ├── wlasl_dataloader.py
│   ├── wlasl_train.py
│   ├── wlasl_evaluate.py
│   ├── frame_predictions.py
│   ├── temporal_aggregation.py
│   ├── hmm_wordLevel.py
│
├── data/              # (ignored in git)
├── models/            # trained models
├── results/           # outputs
└── README.md

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Preprocess Dataset
python src/wlasl_preprocess.py

### 3. Train CNN Model
python src/wlasl_train.py

### 4. Evaluate Model
python src/wlasl_evaluate.py

### 5. Generate Frame Predictions
python src/frame_predictions.py

### 6. Run HMM Decoding
python src/hmm_wordLevel.py


## Live Demo
* OpenCV-based webcam demo for gesture recognition
* Demonstrates real-time prediction capability

## Notes

* Dataset files are not included due to size
* Ensure correct dataset structure before running
* Results may vary depending on hardware and dataset size

## Future Work
* Multi-word sequence recognition (context modeling)
* Improved HMM transition learning
* Integration into a real-time application
* GUI enhancements

## Technologies Used
* Python
* PyTorch
* OpenCV
* NumPy
* scikit-learn

## Conclusion

This project demonstrates how combining CNN-based visual recognition with HMM-based probabilistic decoding can create a robust and uncertainty-aware sign language recognition system.