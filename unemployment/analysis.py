iimport pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv')

#exploring dataset
print(df.head())

#summary stats
print(df.describe())

#checking for any missing values
print(df.isnull().sum())

# Removing leading and trailing spaces in the 'Date' column
df['Date'] = df['Date'].str.strip()

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Line plot of unemployment rate over time for a specific region (e.g., ' Uttar Pradesh')
plt.figure(figsize=(10, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate(%)', data=df[df['Region'] == 'Uttar Pradesh'])
plt.title('Unemployment Rate Over Time in Uttar Pradesh')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.show()
