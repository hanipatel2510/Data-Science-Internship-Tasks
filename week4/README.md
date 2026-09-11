# Week 4 Task: Supervised Learning Model Implementation and Predictive Classification

## 📌 Project Overview
This project presents an end-to-end Supervised Machine Learning pipeline developed using Python and Scikit-learn on the cleaned Titanic passenger survival dataset. The primary objective is to formulate, train, and evaluate a binary classification system that predicts passenger survival outcomes (`Survived`: 0 = Did Not Survive, 1 = Survived) based on demographic profiles, ticket metadata, and socio-economic indicators. 

The complete lifecycle of this task encompasses rigorous data sanitation, categorical encoding, stratified data partitioning, baseline and ensemble model training, 5-fold cross-validation, and diagnostic performance evaluation.

---

## 🎯 Key Objectives
- Formulate a clear binary classification framework on public passenger data.
- Implement systematic data preprocessing to prevent data leakage and noise.
- Implement and benchmark two distinct supervised algorithms: **Logistic Regression** (baseline) and **Random Forest Classifier** (ensemble model).
- Perform 5-Fold Stratified Cross-Validation to assess stability and prevent optimistic overfitting.
- Evaluate model generalization using Accuracy, Precision, Recall, F1-Score, and ROC-AUC curves.
- Extract and interpret feature importance rankings to derive domain insights.

---

## ⚙️ End-to-End Methodology & Pipeline

### 1. Data Cleaning & Feature Selection
- Analyzed all attributes and excluded non-predictive, high-cardinality identifier columns such as `PassengerId`, `Name`, `Ticket`, and `Cabin`. This step eliminates noise and prevents models from memorizing specific record tokens.
- Verified missing value imputation strategies established in earlier pipeline stages, ensuring numerical features (`Age`, `Fare`) and categorical values (`Embarked`) are clean and fully populated.

### 2. Feature Engineering & Categorical Encoding
- Categorical features (`Sex` and `Embarked`) were converted into numeric indicator representations using One-Hot Dummy Encoding (`pd.get_dummies(drop_first=True)`).
- The `drop_first=True` parameter was explicitly applied to prevent multicollinearity and dummy variable traps.

### 3. Stratified Data Partitioning
- The preprocessed dataset was split into an **80% training set** and a **20% testing set**.
- Stratified sampling (`stratify=y`, `random_state=42`) was enforced to maintain identical distributions of survived vs. non-survived classes in both splits.

### 4. Supervised Model Architecture & Rationale
- **Logistic Regression (Baseline Model):** Selected as a linear benchmark to evaluate classification boundaries under logistic loss, offering high interpretability and computational speed.
- **Random Forest Classifier (Proposed Model):** Selected as the primary non-linear ensemble technique. By combining multiple decision trees trained on bootstrap samples (bagging) with random feature subspace sampling, it handles complex interaction terms, remains resilient against outliers, and generates relative feature importances.

### 5. Cross-Validation & Diagnostics
- Evaluated model variance across the training split using **5-Fold Stratified Cross-Validation** (`cv=5`, `StratifiedKFold`).
- Generated diagnostic evaluation artifacts including Confusion Matrices, ROC-AUC comparison curves, cross-validation bar charts, and feature importance rankings.

---

## 📊 Performance Comparison & Evaluation Metrics

Both trained models were evaluated on the held-out 20% test partition to measure true generalization performance:

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Precision | Recall | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 78.4% | 79.3% | 0.76 | 0.68 | 0.84 |
| **Random Forest (Proposed Ensemble)** | **82.8%**[cite: 4] | **80.4%** | **0.84** | **0.61**| **0.84**|

---

## 🔍 Key Insights & Domain Interpretation

- **Dominance of Gender:** Feature importance analysis confirmed that gender (`Sex_male`) was by far the single most decisive factor influencing passenger survival. Female passengers had significantly higher survival probabilities, strongly corroborating the historical maritime evacuation mandate of "women and children first".
- **Socio-Economic Stratification:** Passenger class (`Pclass`) and ticket price (`Fare`) emerged as the second and third most critical factors[cite: 2, 4]. Higher socio-economic standing (first-class accommodations) afforded significantly higher survival odds compared to third-class passengers.
- **Precision Advantage:** The Random Forest Classifier demonstrated higher precision (0.84), which minimized false-positive errors and produced more reliable positive predictions on unseen data.

---

## 📁 Repository Structure

```text
week4-Task/
│
├── dataset/
│   └── cleaned_titanic.csv                 # Cleaned dataset source
│
├── graphs/
│   ├── 01_confusion_matrix.png            # Random Forest confusion matrix heatmap
│   ├── 02_roc_curve.png                   # Baseline vs Ensemble ROC-AUC curves
│   ├── 03_feature_importance.png          # Top 10 predictive feature ranking
│   └── 04_cv_comparison.png               # Cross-validation vs test accuracy comparison
│
├── output/
│   ├── classification_report.txt          # Detailed precision, recall, and F1 logs
│   └── model_metrics_comparison.csv       # Tabulated comparative metrics
│
├── titanic_classification.py              # Main Python ML implementation script
├── requirements.txt                       # Project package dependencies
└── README.md                              # Comprehensive task documentation
