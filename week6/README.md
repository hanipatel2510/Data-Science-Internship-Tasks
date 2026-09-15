# Titanic Integrative Capstone Project (Week 6)

A complete end-to-end data science capstone pipeline combining data preprocessing, feature engineering, exploratory data analysis, unsupervised clustering, and supervised machine learning on the Titanic dataset.

---

## What Was Done in Week 6

* **Data Preprocessing & Cleaning:** Loaded the cleaned dataset, handled remaining null values via stratified median imputation, and prepared numerical and categorical vectors.
* **Feature Engineering:** Built composite features including `FamilySize` (`SibSp + Parch + 1`), solitary passenger flag `IsAlone`, and binary encoded gender (`Sex_Code`).
* **Exploratory Data Analysis (EDA):** Generated demographic survival rates by gender and ticket class alongside a Pearson correlation heatmap across all engineered attributes.
* **Unsupervised Clustering (K-Means):** Standardized continuous variables (`Age`, `Fare`, `Pclass`, `FamilySize`) and identified the optimal 3 passenger clusters using the Elbow method (WCSS).
* **Supervised Machine Learning:** Trained and evaluated two predictive classification models (Baseline Logistic Regression and Random Forest Classifier) using a stratified 80/20 train-test split.
* **Model Evaluation & Diagnostics:** Evaluated accuracy, precision, recall, F1-scores, ROC-AUC curves, and generated a test-set confusion matrix.
* **Automated Asset Export:** Scripted automatic generation and local storage of all 5 visualization figures into `graphs/` and metric summaries into `output/`.

---

## Project Structure

```text
week6-Task/
├── graphs/
│   ├── 01_eda_demographics_survival.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_kmeans_elbow_curve.png
│   ├── 04_roc_curves_and_feature_importance.png
│   └── 05_confusion_matrix.png
├── output/
│   ├── classification_report.txt
│   └── capstone_metrics_summary.csv
├── cleaned_titanic.csv
├── titanic_capstone_pipeline.py
├── requirements.txt
├── README.md
└── WEEK_6_TASK_REPORT.docx
```

---

## Machine Learning Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 71.51% | 0.684 | 0.621 | 0.651 | 0.79 |
| **Random Forest Classifier** | **76.02%** | **0.735** | **0.693** | **0.713** | **0.77** |
| **5-Fold Cross-Validation (RF)** | 75.80% (±2.4%) | 0.720 | 0.685 | 0.702 | — |

---

## Generated Visualizations

1. `01_eda_demographics_survival.png` — Survival rate percentages by gender and ticket class.
2. `02_correlation_heatmap.png` — Heatmap displaying Pearson correlation coefficients across key variables.
3. `03_kmeans_elbow_curve.png` — Sum of squared distances (Inertia) identifying the elbow inflection point at $k = 3$.
4. `04_roc_curves_and_feature_importance.png` — ROC curve comparison and top predictive feature weights.
5. `05_confusion_matrix.png` — Visualized confusion matrix for Random Forest predictions on the test set.

---

## Setup and Execution

1. Navigate to the project directory:
   ```bash
   cd week6-Task
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the complete pipeline script:
   ```bash
   python titanic_capstone_pipeline.py
   ```
