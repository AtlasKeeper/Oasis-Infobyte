import pandas as pd  

#loading dataset
file_path = 'car data.csv'
df = pd.read_csv(file_path)

#checking basic info
print("Info about the dataset:")
print(df.info())

#checking first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

#checking for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

#checking for unique values in each column
print("\nUnique values in each column:")
for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")