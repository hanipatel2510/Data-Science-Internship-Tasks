import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset

df = pd.read_csv("train.csv")
print("Dataset loaded successfully\n")

# Basic information about the dataset

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# Checking missing values

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Missing values visualization

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
plt.figure(figsize=(7, 5))
sns.barplot(x=missing_values.index,y=missing_values.values)
plt.title("Missing Values Before Data Cleaning")
plt.xlabel("Columns")
plt.ylabel("Number of Missing Values")
plt.savefig("missing_values.png", bbox_inches="tight")
plt.show()

# Checking duplicate rows

print("\nDuplicate rows:")
print(df.duplicated().sum())

# Removing duplicates

df = df.drop_duplicates()

# Handling missing values
# Filling missing Age values with median

df["Age"] = df["Age"].fillna(df["Age"].median())

# Filling missing Embarked values with mode

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Removing Cabin column because it has many missing values

df = df.drop("Cabin", axis=1)

# Cleaning text columns

df["Name"] = df["Name"].str.strip()
df["Sex"] = df["Sex"].str.strip().str.lower()
df["Embarked"] = df["Embarked"].str.strip().str.upper()

# Checking invalid values

print("\nInvalid Age values:")
print((df["Age"] < 0).sum())
print("\nInvalid Fare values:")
print((df["Fare"] < 0).sum())

# Checking outliers in Fare column

plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Fare"])
plt.title("Fare Outliers")
plt.savefig("fare_outliers.png", bbox_inches="tight")
plt.show()

# Handling outliers using IQR method

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Capping the Fare outliers

df["Fare"] = np.where(df["Fare"] > upper_limit,upper_limit,df["Fare"])

# Checking the cleaned dataset

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nFinal dataset shape:")
print(df.shape)

# Saving cleaned dataset

df.to_csv("cleaned_titanic.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned dataset saved as cleaned_titanic.csv")