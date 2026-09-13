# 🚢 Week 5: Deep Learning Application in Data Science
**Artificial Neural Network (ANN) for Titanic Passenger Survival Prediction**

---

## 📌 Project Overview
This project implements an end-to-end Deep Learning classification pipeline using **TensorFlow & Keras**. The objective is to design, train, regularize, and evaluate a Deep Feedforward Neural Network (Artificial Neural Network / Multi-Layer Perceptron) to predict binary survival outcomes on the Titanic dataset.

---

## 🎯 Problem Statement
The objective is to formulate a binary classification model that accurately predicts passenger survival:
- **0:** Did Not Survive
- **1:** Survived

Input features include demographic attributes, ticket socio-economic class, fare pricing, and engineered familial attributes.

---

## 🧠 Neural Network Architecture

A customized Multi-Layer Perceptron (MLP) architecture was designed with explicit regularization layers to prevent overfitting on tabular data:

| Layer | Layer Type | Units / Hyperparameters | Activation | Regularization Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `keras.Input` | Input Dimensions (8 features) | - | Standardized tabular input features |
| **Dense 1** | Fully Connected | 64 Neurons | ReLU | Feature extraction + L2 Regularizer (0.001) |
| **BatchNorm 1**| Batch Normalization | - | - | Stabilizes feature activations across mini-batches |
| **Dropout 1** | Dropout Layer | Rate = 0.30 (30%) | - | Mitigates neuron co-adaptation |
| **Dense 2** | Fully Connected | 32 Neurons | ReLU | Non-linear compression + L2 Regularizer (0.001) |
| **Dropout 2** | Dropout Layer | Rate = 0.20 (20%) | - | Controls representation variance |
| **Dense 3** | Fully Connected | 16 Neurons | ReLU | Feature refinement layer |
| **Output** | Dense (Output) | 1 Neuron | Sigmoid | Outputs survival probability [0, 1] |

### Training Hyperparameters:
- **Optimizer:** Adam Optimizer (Learning Rate = 0.001)
- **Loss Function:** Binary Cross-Entropy (`binary_crossentropy`)
- **Batch Size:** 32 samples per mini-batch
- **Epochs:** Up to 100 epochs
- **Early Stopping:** Monitored `val_loss`, Patience = 15, `restore_best_weights = True`

---

## 📂 Repository Structure

```text
week5-Task/
│
├── cleaned_titanic.csv          # Preprocessed Titanic dataset
├── titanic_deep_learning.py     # Deep Learning pipeline script
├── requirements.txt             # Python library dependencies
├── README.md                    # Project documentation
│
├── graphs/                      # Visual artifacts generated automatically
│   ├── 01_training_history.png  # Loss and accuracy curves across epochs
│   ├── 02_confusion_matrix.png  # Confusion matrix heatmap
│   └── 03_roc_curve.png         # Receiver Operating Characteristic (ROC) curve
│
└── output/                      # Evaluation reports & architecture summaries
    ├── classification_report.txt# Precision, Recall, F1-Score breakdown
    ├── deep_learning_metrics.csv# Quantitative metrics summary
    └── model_architecture.txt   # Model layer-by-layer summary
```

---

## ⚙️ Installation & Requirements

Install all required Python packages:

```bash
pip install tensorflow keras pandas numpy scikit-learn matplotlib seaborn
```

---

## 🚀 Execution Guide

Run the main Python script:

```bash
python titanic_deep_learning.py
```

The script automatically:
1. Creates `graphs/` and `output/` directories.
2. Scales and prepares the tabular dataset.
3. Compiles, trains, and validates the Neural Network with early stopping.
4. Generates and saves all evaluation plots and metrics.

---

## 📊 Experimental Results & Evaluation

Evaluated on the unseen holdout test dataset (20% split, 179 samples):

- **Test Accuracy:** 81.01%
- **Precision:** 0.8723
- **Recall:** 0.5942
- **F1-Score:** 0.7069
- **ROC-AUC Score:** 0.8518

### Visual Outputs Saved:
- `01_training_history.png`: Compares Training Loss vs. Validation Loss to verify convergence.
- `02_confusion_matrix.png`: Counts True Positives, True Negatives, False Positives, and False Negatives.
- `03_roc_curve.png`: Evaluates classification discrimination quality across varying thresholds.

---

## 💡 Key Challenges & Mitigation Strategies

1. **Overfitting on Tabular Data:**
   - *Challenge:* Dense neural networks easily memorize small tabular datasets.
   - *Mitigation:* Implemented Dropout (0.3 and 0.2), L2 Regularization, and Early Stopping with weight restoration.

2. **Internal Covariate Shift:**
   - *Challenge:* Unstable gradient flow during training.
   - *Mitigation:* Applied Batch Normalization directly after the primary dense layer.

3. **Feature Scaling Disparity:**
   - *Challenge:* Unscaled features distort backpropagation updates.
   - *Mitigation:* Normalized all continuous numerical features using `StandardScaler`.

