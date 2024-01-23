import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='darkgrid')

#loading the dataset
file_path = 'Advertising.csv'
df = pd.read_csv(file_path)

#basic info
print("Dataset Overview:")
print(df.info())

#display the first few rows
print(df.head())

#summary statistics
print(df.describe())

#checking for any missing values
print(df.isnull().sum())

#visualizing the relationship between features and the target variable
sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.suptitle('Pairplot of Sales with Advertising Channels', y=1.02)
plt.show()

#correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#distribution of sales
plt.figure(figsize=(10, 5))
sns.histplot(df['Sales'], bins=20, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()