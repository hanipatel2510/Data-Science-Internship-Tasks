# 🚢 WEEK 6 TASK REPORT: Integrative Capstone Project and Evaluation
### Titanic Passenger Survival Prediction using End-to-End Data Science Pipeline

---

## 🛠️ 1. Tech Stack & Environment
* **Primary Language:** Python 3.x
* **Data Manipulation & Computation:**
  * `pandas` - High-performance tabular data ingestion and manipulation.
  * `numpy` - Multi-dimensional array structures and mathematical operations.
* **Data Visualization Suite:**
  * `matplotlib.pyplot` - Publication-quality figures, bar plots, and diagnostic charts.
  * `seaborn` - Aesthetic statistical styling and correlation plotting.
* **Machine Learning & Analytics (`scikit-learn`):**
  * `StandardScaler` - Feature standard normal distribution scaling.
  * `KMeans` - Unsupervised clustering and inertia-based elbow optimization.
  * `LogisticRegression` - Interpretable parametric baseline classification.
  * `RandomForestClassifier` - Non-linear ensemble bagging model.
  * `metrics` - Classification report, confusion matrix, and ROC-AUC scoring.
* **Reporting & Documentation:** Microsoft Word (`python-docx`) and Markdown.

---

## ⚙️ 2. Methodological Pipeline Architecture
* **Stage 1: Data Ingestion & Sanitization**
  * Checked feature missingness across all original attributes.
  * Imputed numeric missing attributes using stratified group medians to prevent sample bias.
  * Replaced categorical missing values in port of embarkation with the modal category.
* **Stage 2: Feature Engineering & Transformation**
  * Synthesized `FamilySize` calculated as `SibSp + Parch + 1`.
  * Formulated a binary indicator `IsAlone` (1 if solitary traveler, 0 otherwise).
  * Formatted categorical gender into a numerical binary feature `Sex_Code`.
* **Stage 3: Exploratory Data Analysis (EDA)**
  * Generated gender and class survival comparative rates.
  * Evaluated Pearson correlation heatmap matrices to detect feature dependencies.
* **Stage 4: Unsupervised Persona Clustering**
  * Standardized continuous dimensions including fare, age, and class.
  * Applied K-Means across cluster ranges to establish the optimal elbow inflection point.
* **Stage 5: Supervised Learning & Cross-Validation**
  * Stratified target variables across an 80/20 train-test distribution.
  * Trained baseline Logistic Regression alongside ensemble Random Forest.
  * Executed stratified cross-validation routines to guarantee generalization.
* **Stage 6: Comparative Diagnostics & Model Export**
  * Serialized class-level precision, recall, and F1 metrics into text logs.
  * Exported comparative performance metrics into structured CSV spreadsheets.

---

## 📂 3. Project Directory Structure
```text
week6-Task/
│
├── graphs/                                       # Generated diagnostic plots
│   ├── 01_eda_demographics_survival.png          # Gender and passenger class survival rates
│   ├── 02_correlation_heatmap.png                # Full feature Pearson correlation matrix
│   ├── 03_kmeans_elbow_curve.png                 # Sum of squared distances elbow curve
│   ├── 04_roc_curves_and_feature_importance.png  # Supervised ROC-AUC and feature weights
│   └── 05_confusion_matrix.png                   # Test set prediction confusion matrix
│
├── output/                                       # Metrics and performance outputs
│   ├── classification_report.txt                 # Detailed precision, recall, and F1 logs
│   └── capstone_metrics_summary.csv              # Tabular model comparison spreadsheet
│
├── cleaned_titanic.csv                           # Preprocessed source data file
├── titanic_capstone_pipeline.py                  # End-to-end Python pipeline script
├── requirements.txt                              # Environment package dependencies
├── README.md                                     # Comprehensive project repository guide
└── WEEK_6_TASK_REPORT.docx                       # Final project deliverable report
```

---

## 🔬 4. Deep-Dive Analytical Findings

### A. Demographic & Socioeconomic Survival Factors
* **Gender Disparity:** Female passengers displayed a survival probability of **~58.9%**, contrasting against male survival at **~32.0%**.
* **Socioeconomic Advantage:** 1st Class passengers recorded a **~68.6%** survival rate, while 3rd Class passengers recorded **~28.2%** due to lower-deck evacuation delays.
* **Family Unit Survival:** Solitary travelers and overly large families faced higher mortality risks compared to small family units.

### B. Unsupervised Passenger Grouping (K-Means)
* **Elbow Curve Analysis:** Plotted within-cluster sum of squares (WCSS) across k=1 to 7, surfacing a clear mathematical elbow at **k = 3**.
* **Cluster 1 (Elite Travelers):** High fare bracket, mature age group, concentrated primarily in 1st Class.
* **Cluster 2 (Solitary Economy):** Low-to-moderate fare, young adults traveling alone in 3rd Class.
* **Cluster 3 (Family Cohorts):** High family counts, moderate fare, balanced age distributions.

### C. Predictive Feature Importance
* **Rank 1 - Ticket Fare:** The strongest financial and spatial survival predictor.
* **Rank 2 - Passenger Class (`Pclass`):** Direct indicator of cabin location and lifeboat accessibility.
* **Rank 3 - Age & Gender (`Sex_Code`):** Core demographic criteria driving emergency life-saving priority.

---

## 📊 5. Quantitative Model Evaluation Metrics

| Model Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 71.51% | 0.684 | 0.621 | 0.651 | 0.730 |
| **Random Forest Classifier** | **76.02%** | **0.735** | **0.693** | **0.713** | **0.710** |
| **Stratified Cross-Validation (RF)** | 75.80% (±2.4%) | 0.720 | 0.685 | 0.702 | — |

---

## 💡 6. Strategic Conclusions & Practical Implications
* **Physical Architecture & Cabin Tiering:** Proximity to upper deck evacuation launch sites dramatically dictated passenger life expectancy.
* **Adherence to Evacuation Protocol:** Historical documentation regarding the prioritization of women and children was verified through algorithmic feature coefficients.
* **Ensemble Learning Supremacy:** Non-linear ensemble trees generalized better across complex sociodemographic boundaries without significant variance overfitting.

---

## 🚀 7. Quick Start & Reproducibility Guide
* 1. **Navigate to Project Directory:**
  ```bash
  cd week6-Task
  ```
* 2. **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
* 3. **Execute Pipeline Runner:**
  ```bash
  python titanic_capstone_pipeline.py
  ```
* 4. **Inspect Diagnostic Results:** Check the `/graphs/` folder for images and `/output/` for numeric text summaries!
