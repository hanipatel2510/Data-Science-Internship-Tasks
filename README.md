# Titanic Data Cleaning and Preprocessing

## Week 1 Internship Task

### About the Project

This project focuses on cleaning and preprocessing the Titanic dataset using Python. The main aim was to check the dataset, find missing values, remove unnecessary data, check duplicate records, and identify outliers.

### Dataset

The Titanic dataset contains information about passengers such as age, gender, passenger class, ticket, fare, cabin, and survival status.

The original dataset contains 891 rows and 12 columns.

### Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Steps Performed

- Loaded the Titanic dataset.
- Checked the first five rows of the dataset.
- Checked the dataset shape and information.
- Checked missing values.
- Created a graph to visualize missing values.
- Checked duplicate records.
- Filled missing Age values using the median.
- Filled missing Embarked values using the mode.
- Removed the Cabin column because most values were missing.
- Cleaned text columns.
- Checked invalid Age and Fare values.
- Used a boxplot to identify Fare outliers.
- Used the IQR method to handle outliers.
- Saved the cleaned dataset.

### Project Files

- `train.csv` - Original dataset
- `titanic_preprocessing.py` - Python preprocessing code
- `cleaned_titanic.csv` - Cleaned dataset
- `requirements.txt` - Required Python libraries

### Output

After cleaning and preprocessing, the final dataset was saved as `cleaned_titanic.csv`.
