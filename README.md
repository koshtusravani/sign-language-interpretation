# Uncertainty-Aware Contextual Recognition of Sign Language Sequences

## Project Description

This project explores how deep learning and probabilistic sequence modeling can improve sign language recognition by incorporating contextual information across gesture sequences. A convolutional neural network (CNN) is used to classify individual American Sign Language (ASL) gestures from images, and a Hidden Markov Model (HMM) is used to refine predictions across sequences using probabilistic decoding.

The goal of the project is to compare a frame-based CNN classifier with a context-aware probabilistic sequence model and analyze how contextual reasoning can improve recognition performance when predictions are uncertain.

---

## Project Structure

sign-language-interpretation/

src/  
- preprocess.py 
- data_loader.py
- cnn_training.py
- evaluate.py 
- generate_predictions.py 
- build_sequences.py 
- hmm_decoding.py 
- compare_predictions.py 
- compare_sequences.py 

data/  
- raw/ — Original ASL dataset (not tracked in Git)  
- processed/ — Processed train/validation/test dataset  
- metadata/  
  - class_names.txt — List of gesture classes  

results/ — Evaluation outputs and experiment results  

requirements.txt — Python dependencies  
.gitignore — Ignored files and directories  
README.md — Project documentation  

---

## How to Run the Pipeline

### 1. Install dependencies

pip install -r requirements.txt

### 2. Place the dataset

Download the ASL alphabet dataset and place it inside:

data/raw/

dataset link: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

### 3. Preprocess the dataset

python src/preprocess.py

This step creates the processed dataset used for training and evaluation.

### 4. Data Loader

The dataset is loaded using data_loader.py.  
This file is not usually run directly, but it is used internally by the training and evaluation scripts to load the dataset.

(Optional test)

python src/data_loader.py

### 5. Train the CNN model

python src/cnn_training.py

This trains the baseline ResNet18 model and saves training logs.

### 6. Evaluate the model

python src/evaluate.py

This generates evaluation metrics such as accuracy, confusion matrix, and classification report.

### 7. Generate CNN prediction probabilities

python src/generate_predictions.py

### 8. Build sequence experiments

python src/build_sequences.py

### 9. Run HMM decoding

python src/hmm_decoding.py

### 10. Compare predictions

python src/compare_sequences.py

All outputs and evaluation results will be saved in the results/ directory.