import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 1. Output directories creation
os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)

print("=" * 60)
print("WEEK 6: INTEGRATIVE CAPSTONE PROJECT PIPELINE")
print("=" * 60)

# 2. Data Loading (Auto-detect exact location)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "cleaned_titanic.csv")

# Check fallback paths if file is not found in the script directory
if not os.path.exists(csv_path):
    csv_path = "cleaned_titanic.csv"

if not os.path.exists(csv_path):
    csv_path = os.path.join(base_dir, "..", "week1_Titanic_Data_Preprocessing", "cleaned_titanic.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded dataset from {csv_path} with shape: {df.shape}")
else:
    raise FileNotFoundError(f"Dataset file not found! Please verify file location: {csv_path}")

# Ensure all outputs and figures are saved inside the current working directory
os.chdir(base_dir)
os.makedirs("graphs", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 3. Feature Engineering & Preparation
if "FamilySize" not in df.columns and "SibSp" in df.columns and "Parch" in df.columns:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

if "IsAlone" not in df.columns and "FamilySize" in df.columns:
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

if "Sex_Code" not in df.columns and "Sex" in df.columns:
    df["Sex_Code"] = (df["Sex"].str.lower() == "female").astype(int)

# 4. Exploratory Data Analysis & Graph Generation
print("[+] Generating EDA Visualizations...")
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# Graph 1: Demographics Survival Rate
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)
if "Sex" in df.columns:
    surv_gender = df.groupby("Sex")["Survived"].mean() * 100
    bars1 = axes[0].bar(surv_gender.index, surv_gender.values, color=["#e74c3c", "#2980b9"], width=0.45, edgecolor="black")
    axes[0].set_title("Survival Rate by Gender (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Survival Rate (%)")
    axes[0].set_ylim(0, 100)
    for b in bars1:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontweight="bold")

if "Pclass" in df.columns:
    surv_class = df.groupby("Pclass")["Survived"].mean() * 100
    bars2 = axes[1].bar([f"Class {c}" for c in surv_class.index], surv_class.values, color=["#27ae60", "#f39c12", "#8e44ad"], width=0.45, edgecolor="black")
    axes[1].set_title("Survival Rate by Ticket Class (%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].set_ylim(0, 100)
    for b in bars2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("graphs/01_eda_demographics_survival.png")
plt.close()

# Graph 2: Correlation Heatmap
corr_cols = [c for c in ["Survived", "Pclass", "Age", "Fare", "FamilySize", "Sex_Code"] if c in df.columns]
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(cax)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols, rotation=45, ha="left", fontweight="bold")
ax.set_yticklabels(corr_cols, fontweight="bold")
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(corr.iloc[i, j]) > 0.4 else "black", fontweight="bold")
plt.title("Correlation Matrix Heatmap", fontsize=12, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("graphs/02_correlation_heatmap.png")
plt.close()

# 5. Unsupervised Learning: K-Means Clustering
print("[+] Running Unsupervised K-Means Clustering...")
cluster_features = [c for c in ["Age", "Fare", "Pclass", "FamilySize"] if c in df.columns]
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(df[cluster_features].dropna())

inertias = []
k_range = range(1, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
ax.plot(k_range, inertias, marker="o", color="#2c3e50", lw=2, markersize=6, markerfacecolor="#e74c3c")
ax.axvline(x=3, color="#e67e22", linestyle="--", label="Elbow Point (k=3)")
ax.set_title("K-Means Clustering: Elbow Curve for Optimal K", fontsize=11, fontweight="bold")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("Inertia (Sum of Squared Distances)")
ax.legend()
plt.tight_layout()
plt.savefig("graphs/03_kmeans_elbow_curve.png")
plt.close()

# 6. Supervised Learning: Model Training & Evaluation
print("[+] Training Supervised Classification Models...")
feature_cols = [c for c in ["Pclass", "Age", "Fare", "FamilySize", "IsAlone", "Sex_Code"] if c in df.columns]
X = df[feature_cols].fillna(df[feature_cols].median())
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_probs = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]

# Graph 4: ROC Curves & Feature Importance
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=200)
axes[0].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc(fpr_lr, tpr_lr):.2f})", color="#3498db", lw=2)
axes[0].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})", color="#2ecc71", lw=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set_title("Supervised ROC-AUC Comparison", fontsize=11, fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")

importances = rf.feature_importances_
idx = np.argsort(importances)
axes[1].barh(range(len(idx)), importances[idx], color="#8e44ad", edgecolor="black")
axes[1].set_yticks(range(len(idx)))
axes[1].set_yticklabels([feature_cols[i] for i in idx], fontweight="bold")
axes[1].set_xlabel("Importance Score")
axes[1].set_title("Random Forest Feature Importance", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/04_roc_curves_and_feature_importance.png")
plt.close()

# Graph 5: Confusion Matrix
cm = confusion_matrix(y_test, rf_preds)
fig, ax = plt.subplots(figsize=(5, 4), dpi=200)
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred: Perished", "Pred: Survived"], fontweight="bold")
ax.set_yticklabels(["Actual: Perished", "Actual: Survived"], fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > np.max(cm) / 2 else "black", fontsize=13, fontweight="bold")
plt.title("Random Forest Confusion Matrix", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("graphs/05_confusion_matrix.png")
plt.close()

# 7. Save Text Output & Metrics Summary
with open("output/classification_report.txt", "w") as f:
    f.write("=== LOGISTIC REGRESSION ===\n")
    f.write(classification_report(y_test, lr_preds))
    f.write("\n=== RANDOM FOREST CLASSIFIER ===\n")
    f.write(classification_report(y_test, rf_preds))

metrics_df = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": f"{accuracy_score(y_test, lr_preds)*100:.2f}%",
        "Precision": f"{precision_score(y_test, lr_preds):.3f}",
        "Recall": f"{recall_score(y_test, lr_preds):.3f}",
        "F1-Score": f"{f1_score(y_test, lr_preds):.3f}",
        "ROC-AUC": f"{auc(fpr_lr, tpr_lr):.3f}",
    },
    {
        "Model": "Random Forest",
        "Accuracy": f"{accuracy_score(y_test, rf_preds)*100:.2f}%",
        "Precision": f"{precision_score(y_test, rf_preds):.3f}",
        "Recall": f"{recall_score(y_test, rf_preds):.3f}",
        "F1-Score": f"{f1_score(y_test, rf_preds):.3f}",
        "ROC-AUC": f"{auc(fpr_rf, tpr_rf):.3f}",
    },
])
metrics_df.to_csv("output/capstone_metrics_summary.csv", index=False)

print("[✓] Pipeline finished successfully!")
print("    -> 5 Graphs saved to 'graphs/' folder.")
print("    -> Metrics & reports saved to 'output/' folder.")