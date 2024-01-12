import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets with absolute paths
df_india = pd.read_csv(r'A:\Code\Projects\Oasis InfoByte\Oasis-Infobyte\unemployment\Unemployment in India.csv')
df_up_to_11_2020 = pd.read_csv(r'A:\Code\Projects\Oasis InfoByte\Oasis-Infobyte\unemployment\Unemployment_Rate_upto_11_2020.csv')

# Exploring the dataset
print("Columns in India dataset:", df_india.columns)
print("Columns in up to Nov 2020 dataset:", df_up_to_11_2020.columns)

# Checking for missing values in the datasets
print("Missing values in India dataset:\n", df_india.isnull().sum())
print("Missing values in up to Nov 2020 dataset:\n", df_up_to_11_2020.isnull().sum())

# Removing rows with missing values
df_india = df_india.dropna()
df_up_to_11_2020 = df_up_to_11_2020.dropna()

print(df_up_to_11_2020.head())
print(df_india.head())

# Mergining datasets
merged_df = pd.concat([df_india, df_up_to_11_2020], ignore_index=True, sort=False)

print(merged_df.head())
