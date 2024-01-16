import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace 'your_file.csv' with the actual file path
file_path = 'Unemployment_Rate_upto_11_2020.csv'

# Read the CSV file into a pandas DataFrame
unemployment_data = pd.read_csv(file_path)

# Display basic information about the DataFrame
print("Data Overview:")
print(unemployment_data.info())

# Display the first few rows of the DataFrame to understand the data structure
print("\nFirst 20 rows of the data:")
print(unemployment_data.head(20))

print("Column Names:")
print(unemployment_data.columns)

# Remove leading and trailing spaces from column names
unemployment_data.columns = unemployment_data.columns.str.strip()

# 1. Average Unemployment Rate
average_unemployment_rate = unemployment_data['Estimated Unemployment Rate (%)'].mean()
print(f"\nAverage Unemployment Rate: {average_unemployment_rate:.2f}%")

# 2. Maximum Unemployment Rate
max_unemployment_row = unemployment_data.loc[unemployment_data['Estimated Unemployment Rate (%)'].idxmax()]
print(f"\nMaximum Unemployment Rate:")
print(max_unemployment_row)

# 3. Region-wise Analysis
region_wise_average = unemployment_data.groupby('Region')['Estimated Unemployment Rate (%)'].mean()
print("\nRegion-wise Average Unemployment Rate:")
print(region_wise_average)

# 4. Temporal Analysis
temporal_analysis = unemployment_data.groupby('Date')['Estimated Unemployment Rate (%)'].mean()
print("\nTemporal Analysis - Average Unemployment Rate over Time:")
print(temporal_analysis)

# Visualizations
# 1. Average and Maximum Unemployment Rate - Single Bar
plt.figure(num='Average_Max_Unemployment_Rate', figsize=(10, 6))
plt.bar(['Average Unemployment Rate', 'Maximum Unemployment Rate'], [average_unemployment_rate, max_unemployment_row['Estimated Unemployment Rate (%)']], color=['blue', 'red'])
plt.title('Average and Maximum Unemployment Rate')
plt.ylabel('Unemployment Rate (%)')
plt.show()

# 2. Region-wise Analysis - Bar Chart
plt.figure(num='Region_wise_Avg_Unemployment_Rate', figsize=(14, 8))
region_wise_average.plot(kind='bar', color='green')
plt.title('Region-wise Average Unemployment Rate')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.show()

# 3. Temporal Analysis - Line Chart
plt.figure(num='Temporal_Analysis_Avg_Unemployment_Rate', figsize=(14, 8))
temporal_analysis.plot(kind='line', marker='o', color='purple')
plt.title('Temporal Analysis - Average Unemployment Rate over Time')
plt.ylabel('Average Unemployment Rate (%)')
plt.xticks(rotation=45, ha='right')
plt.show()

# 5. Correlation Analysis
numeric_columns = unemployment_data.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = unemployment_data[numeric_columns].corr()

# Visualize the correlation matrix
plt.figure(num='Correlation_Matrix', figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 6. Geographical Distribution
plt.figure(num='Geographical_Distribution', figsize=(14, 8))
plt.scatter(unemployment_data['longitude'], unemployment_data['latitude'], c=unemployment_data['Estimated Unemployment Rate (%)'], cmap='coolwarm', s=100, alpha=0.7)
plt.title('Geographical Distribution of Unemployment Rates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.colorbar(label='Unemployment Rate (%)')
plt.show()

# 7. Frequency Analysis
plt.figure(num='Frequency_Distribution', figsize=(10, 6))
unemployment_data['Frequency'].value_counts().sort_index().plot(kind='bar', color='orange')
plt.title('Frequency Distribution of Unemployment Data')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.show()

# Additional Analyses
# 8. Outlier Detection - Box Plot
plt.figure(num='Outlier_Detection_Box_Plot', figsize=(10, 6))
sns.boxplot(x=unemployment_data['Estimated Unemployment Rate (%)'])
plt.title('Box Plot of Estimated Unemployment Rate')
plt.show()

# 9. Distribution of Employed
plt.figure(num='Distribution_of_Employed', figsize=(12, 8))
sns.histplot(unemployment_data['Estimated Employed'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Estimated Employed')
plt.xlabel('Estimated Employed')
plt.ylabel('Frequency')
plt.show()

# 10. Distribution of Labor Participation Rate
plt.figure(num='Distribution_of_Labor_Participation_Rate', figsize=(12, 8))
sns.histplot(unemployment_data['Estimated Labour Participation Rate (%)'], kde=True, color='salmon', bins=20)
plt.title('Distribution of Estimated Labour Participation Rate')
plt.xlabel('Estimated Labour Participation Rate (%)')
plt.ylabel('Frequency')
plt.show()
