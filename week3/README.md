# Week 3 – Unsupervised Learning and Clustering Analysis

This project performs **unsupervised learning and clustering analysis** on the cleaned Titanic dataset used in the previous weeks of the internship.

## Objective

The main objective is to group Titanic passengers into meaningful clusters using **K-Means clustering** and analyze the characteristics of each group.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
week3-Task/
│
├── graphs/
│   ├── 01_elbow_method.png
│   ├── 02_silhouette_scores.png
│   ├── 03_cluster_visualization.png
│   ├── 04_cluster_profile_heatmap.png
│   └── 05_cluster_sizes.png
│
├── output/
│   ├── clustering_summary.txt
│   ├── cluster_profile.csv
│   ├── silhouette_scores.csv
│   └── titanic_clustered.csv
│
├── titanic_clustering.py
├── README.md
└── Week3_Clustering_Report.docx
```

## Dataset

The project reuses the cleaned Titanic dataset from **Week 2**. The script automatically looks for `cleaned_titanic.csv` inside the Week 3 folder or the Week 2 folder.

## Clustering Method

### K-Means Clustering

K-Means is used to divide passengers into groups with similar characteristics.

The clustering features are selected from:

- Pclass
- Age
- SibSp
- Parch
- Fare
- Sex (encoded numerically when available)

The `Survived` column is intentionally excluded from clustering because it is an outcome variable. This keeps the analysis unsupervised.

## Preprocessing

Before clustering:

1. Numeric features are converted to numeric values.
2. Missing numeric values are filled using the median.
3. The Sex column is encoded as a numeric feature.
4. Features are standardized using `StandardScaler`.

Standardization is important because features such as Fare and Pclass have different scales.

## Choosing the Number of Clusters

Two methods are used:

- **Elbow Method** – compares K-Means inertia for different values of K.
- **Silhouette Score** – measures how well-separated the clusters are.

The K value with the highest silhouette score from the tested range is selected automatically.

## Visualization

The project creates:

1. Elbow Method graph
2. Silhouette Score graph
3. PCA-based cluster scatter plot
4. Cluster profile heatmap
5. Cluster size bar chart

PCA is used only to visualize the high-dimensional clustering result in two dimensions.

## Outputs

The final clustered dataset is saved as:

`output/titanic_clustered.csv`

Additional files contain cluster profiles, silhouette scores, and a text summary.

## How to Run

From the Week 3 folder:

```bash
python titanic_clustering.py
```

Make sure the Week 2 `cleaned_titanic.csv` is available in the expected Week 2 folder.

## Key Insight

The purpose of clustering is not to predict survival. Instead, it discovers passenger segments based on demographic, travel-class, family, and fare-related characteristics. These segments can then be interpreted to understand different passenger groups in the Titanic dataset.
