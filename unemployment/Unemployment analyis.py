import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
# Load the datasets with absolute paths
df_india = pd.read_csv(r'A:\Code\Projects\Oasis InfoByte\Oasis-Infobyte\unemployment\Unemployment in India.csv')
df_up_to_11_2020 = pd.read_csv(r'A:\Code\Projects\Oasis InfoByte\Oasis-Infobyte\unemployment\Unemployment_Rate_upto_11_2020.csv')

#Exploring the dataset
print(df_india.head())
print((df_up_to_11_2020.head()))

#checking for missing values in data set
print("Missing values in India dataset:\n", df_india.isnull().sum())
print("Missing values in up to Nov 2020 dataset:\n", df_up_to_11_2020.isnull().sum())

#removing rows with missing values
df_india = df_india.dropna()
df_up_to_11_2020 = df_up_to_11_2020.dropna()

# Merging datasets based on common identifiers
merged_df = pd.merge(df_india, df_up_to_11_2020, on='Date', how='outer')

#converting 'Date' column to datetime type
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

print(merged_df.head())
