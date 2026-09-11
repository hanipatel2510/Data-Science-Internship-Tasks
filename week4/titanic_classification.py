import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report)

# Create output folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 1. Load Dataset (cleaned_titanic.csv from week2/week3 or current dir)
csv_path = "cleaned_titanic.csv"
if not os.path.exists(csv_path):
    csv_path = "cleaned_titanic.csv"

df = pd.read_csv(csv_path)

# Drop non-predictive or redundant text columns if present
drop_cols = [c for c in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if c in df.columns]
df = df.drop(columns=drop_cols)

# Handle categorical variables (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# 2. Features & Target definition
X = df.drop(columns=['Survived'])
y = df['Survived']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Model Definition: Baseline (Logistic Regression) vs Proposed (Random Forest)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
}

results = []

for name, model in models.items():
    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    results.append({
        "Model": name,
        "CV Mean Accuracy": cv_scores.mean(),
        "Test Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

# Save Metrics
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("output/model_metrics_comparison.csv", index=False)
print("\n--- Model Comparison ---\n", metrics_df)

# Evaluate Best Model (Random Forest)
rf_model = models["Random Forest"]
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Save Classification Report
report = classification_report(y_test, y_pred_rf)
with open("output/classification_report.txt", "w") as f:
    f.write("Random Forest Classification Report\n")
    f.write(report)

# ----------------- Visualizations -----------------

# Graph 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Did Not Survive (0)', 'Survived (1)'],
            yticklabels=['Did Not Survive (0)', 'Survived (1)'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig("graphs/01_confusion_matrix.png", dpi=300)
plt.close()

# Graph 2: ROC Curve Comparison
plt.figure(figsize=(7, 5))
for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probs):.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
plt.title("ROC Curves Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graphs/02_roc_curve.png", dpi=300)
plt.close()

# Graph 3: Feature Importance (Random Forest)
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Relative Importance Score")
plt.tight_layout()
plt.savefig("graphs/03_feature_importance.png", dpi=300)
plt.close()

# Graph 4: Cross-Validation vs Test Accuracy
plt.figure(figsize=(7, 5))
x_axis = np.arange(len(metrics_df))
plt.bar(x_axis - 0.15, metrics_df["CV Mean Accuracy"], width=0.3, label="CV Accuracy", color="#3b82f6")
plt.bar(x_axis + 0.15, metrics_df["Test Accuracy"], width=0.3, label="Test Accuracy", color="#10b981")
plt.xticks(x_axis, metrics_df["Model"])
plt.ylim(0, 1.0)
plt.title("Cross-Validation vs Test Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.legend()
plt.grid(alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig("graphs/04_cv_comparison.png", dpi=300)
plt.close()

print("\nModel training and graph generation completed successfully!")