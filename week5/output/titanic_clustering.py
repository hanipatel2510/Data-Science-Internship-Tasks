# Week 5 - Unsupervised Learning and Clustering Analysis
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# 1. Paths and folders
BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR / "graphs"
OUTPUT_DIR = BASE_DIR / "output"

GRAPH_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Reuse the cleaned Titanic dataset from Week 2.
DATA_PATHS = [
    BASE_DIR / "cleaned_titanic.csv",
    BASE_DIR.parent / "week2-Task" / "cleaned_titanic.csv",
    BASE_DIR.parent / "week2-Task" / "output" / "cleaned_titanic.csv",
]

DATA_PATH = next((path for path in DATA_PATHS if path.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError("cleaned_titanic.csv was not found. Keep the Week 2 cleaned_titanic.csv " "inside week2-Task or copy it into week3-Task.")


# ---------------------------------------------------------
# 2. Load data
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("WEEK 3 - TITANIC CLUSTERING ANALYSIS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ---------------------------------------------------------
# 3. Select features for unsupervised learning
# ---------------------------------------------------------
# Survived is deliberately NOT used because it is the target/outcome
# from the original Titanic dataset. Clustering should discover groups
# without using the survival result.

numeric_features = [col for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"] if col in df.columns]

categorical_features = [col for col in ["Sex"] if col in df.columns]

if not numeric_features:
    raise ValueError("Required numeric Titanic features were not found.")

cluster_df = df[numeric_features + categorical_features].copy()

# Convert numeric columns to numeric values and fill missing values.
for col in numeric_features:
    cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce")
    cluster_df[col] = cluster_df[col].fillna(cluster_df[col].median())

# Encode Sex if it exists.
if "Sex" in cluster_df.columns:
    cluster_df["Sex"] = (
        cluster_df["Sex"]
        .astype(str)
        .str.lower()
        .map({"male": 0, "female": 1})
        .fillna(0.5)
    )

print("\nFeatures used for clustering:")
print(cluster_df.columns.tolist())

print("\nMissing values after preprocessing:")
print(cluster_df.isnull().sum())


# ---------------------------------------------------------
# 4. Standardize features
# ---------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(cluster_df)

print("\nStandardized data shape:", X.shape)


# ---------------------------------------------------------
# 5. Elbow Method - choose number of clusters
# ---------------------------------------------------------
inertias = []
k_values = range(2, 9)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "01_elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 6. Silhouette Score - validate cluster quality
# ---------------------------------------------------------
silhouette_scores = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

score_table = pd.DataFrame({
    "Number_of_Clusters": list(k_values),
    "Silhouette_Score": silhouette_scores
})

print("\nSilhouette scores:")
print(score_table.to_string(index=False))

best_k = int(
    score_table.loc[
        score_table["Silhouette_Score"].idxmax(),
        "Number_of_Clusters"
    ]
)

print("\nSelected number of clusters:", best_k)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_scores, marker="o")
plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(GRAPH_DIR / "02_silhouette_scores.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 7. Apply K-Means clustering
# ---------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

final_silhouette = silhouette_score(X, df["Cluster"])

print("\nFinal K-Means silhouette score:", round(final_silhouette, 4))
print("\nCluster counts:")
print(df["Cluster"].value_counts().sort_index())


# ---------------------------------------------------------
# 8. PCA - reduce dimensions for visualization
# ---------------------------------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

cluster_plot = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": df["Cluster"].astype(str)
})

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=cluster_plot,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="deep",
    s=70
)
plt.title("Titanic Passenger Clusters using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "03_cluster_visualization.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 9. Cluster profile
# ---------------------------------------------------------
profile_columns = [col for col in numeric_features if col in df.columns]

if "Sex" in df.columns:
    # Create a numeric female indicator only for profile interpretation.
    df["Female"] = (
        df["Sex"]
        .astype(str)
        .str.lower()
        .map({"female": 1, "male": 0})
    )
    profile_columns = profile_columns + ["Female"]

cluster_profile = df.groupby("Cluster")[profile_columns].mean().round(2)

print("\nCluster profile:")
print(cluster_profile)

cluster_profile.to_csv(OUTPUT_DIR / "cluster_profile.csv")

plt.figure(figsize=(10, 6))
sns.heatmap(cluster_profile, annot=True, fmt=".2f", cmap="Blues")
plt.title("Cluster Profile - Average Feature Values")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "04_cluster_profile_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 10. Cluster size visualization
# ---------------------------------------------------------
cluster_counts = df["Cluster"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values
)
plt.title("Number of Passengers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(GRAPH_DIR / "05_cluster_sizes.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 11. Save final clustered dataset
# ---------------------------------------------------------
df.to_csv(OUTPUT_DIR / "titanic_clustered.csv", index=False)

score_table.to_csv(OUTPUT_DIR / "silhouette_scores.csv", index=False)

with open(OUTPUT_DIR / "clustering_summary.txt", "w", encoding="utf-8") as file:
    file.write("Titanic K-Means Clustering Summary\n")
    file.write("=" * 40 + "\n")
    file.write(f"Dataset: {DATA_PATH.name}\n")
    file.write(f"Rows: {len(df)}\n")
    file.write(f"Features used: {', '.join(cluster_df.columns)}\n")
    file.write(f"Selected clusters (K): {best_k}\n")
    file.write(f"Final silhouette score: {final_silhouette:.4f}\n\n")
    file.write("Cluster sizes:\n")
    file.write(cluster_counts.to_string())
    file.write("\n\nCluster profile:\n")
    file.write(cluster_profile.to_string())

print("\nFiles created successfully inside:")
print("Graphs :", GRAPH_DIR)
print("Output :", OUTPUT_DIR)

print("\nWeek 3 clustering analysis completed successfully.")
