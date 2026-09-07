import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Current file ki directory aur graphs folder ka path set karein
base_dir = os.path.dirname(os.path.abspath(__file__))
graphs_dir = os.path.join(base_dir, "graphs")

# Create folder for graphs inside week2-Task
os.makedirs(graphs_dir, exist_ok=True)

# Load the dataset
csv_path = os.path.join(base_dir, "cleaned_titanic.csv")
df = pd.read_csv(csv_path)
print("Titanic dataset loaded successfully")

# Dataset overview
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

# Basic statistical analysis
print("\nStatistical summary:")
print(df.describe())

# Survival analysis
print("\nSurvival count:")
print(df["Survived"].value_counts())
print("\nSurvival percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

# Gender analysis
print("\nSurvival by gender:")
print(pd.crosstab(df["Sex"], df["Survived"]))

# Passenger class analysis
print("\nSurvival by passenger class:")
print(pd.crosstab(df["Pclass"], df["Survived"]))

# Average age by survival
print("\nAverage age by survival:")
print(df.groupby("Survived")["Age"].mean())

# Average fare by passenger class
print("\nAverage fare by passenger class:")
print(df.groupby("Pclass")["Fare"].mean())

# Graph 1: Survival count
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Count")
plt.xlabel("Survival Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig(os.path.join(graphs_dir, "survival_count.png"))
plt.show()

# Graph 2: Gender and survival
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival Based on Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.savefig(os.path.join(graphs_dir, "gender_survival.png"))
plt.show()

# Graph 3: Passenger class and survival
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival Based on Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.savefig(os.path.join(graphs_dir, "passenger_class_survival.png"))
plt.show()

# Graph 4: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig(os.path.join(graphs_dir, "age_distribution.png"))
plt.show()

# Graph 5: Fare distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution of Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.savefig(os.path.join(graphs_dir, "fare_distribution.png"))
plt.show()

# Graph 6: Correlation heatmap
numeric_columns = df.select_dtypes(include="number")
plt.figure(figsize=(10, 7))
sns.heatmap(numeric_columns.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(graphs_dir, "correlation_heatmap.png"))
plt.show()

print("\nExploratory Data Analysis completed successfully!")
print("Graphs have been saved in the week2-Task/graphs folder.")