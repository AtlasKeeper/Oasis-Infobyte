import pandas as pd

file_path = 'Unemployment in India.csv'
file_path2 = 'Unemployment_Rate_upto_11_2020.csv'

df = pd.read_csv(file_path)
df2 = pd.read_csv(file_path2)

columns_in_df1 = set(df.columns)
columns_in_df2 = set(df2.columns)

columns_only_in_df1 = columns_in_df1 - columns_in_df2
columns_only_in_df2 = columns_in_df2 - columns_in_df1

print("Columns only in the first DataFrame:", columns_only_in_df1)
print("columns only in the second DataFrame:", columns_only_in_df2)

# Removing leading and trailing spaces in column Names
df.columns = df.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Removing leading and trailing spaces in the 'Date' column
df['Date'] = df['Date'].str.strip()
df2['Date'] = df2['Date'].str.strip()

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', dayfirst=True)
df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y', dayfirst=True)

# Merging the DataFrames on the 'Date' columns
merged_df = pd.merge(df, df2, on='Date', how='inner')

print(merged_df.head())
