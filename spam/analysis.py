import pandas as pd 
from sklearn.model_selection import train_test_split
import os

#Loading the datset
spam_data = pd.read_csv("spam.csv", encoding="latin-1")

#exploring the dataset 
print(spam_data.head())
print(spam_data.info())
print(spam_data.describe())

#checking the distribution of the 'label' column
print(spam_data['v1'].value_counts())

#dropping unnecessary columns
spam_data = spam_data[['v1', 'v2']]

#checking for missing values
print("Missing values in each column")
print(spam_data.isnull().sum())

#spliting the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    spam_data['v2'], spam_data['v1'], test_size=0.2, random_state=42
)

#display the shapes of the training and testing sets
print("\nTraining set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

#saving the cleaned dataset to a separate location 
cleaned_file_path = "cleaned_data/cleaned_spam_data.csv"

#check if the file already exists
if os.path.exists(cleaned_file_path):
    print(f"\nCleaned dataset already exists at: {cleaned_file_path}")
else:
    spam_data.to_csv(cleaned_file_path, index=False)
    print(f"\nCleaned dataset saved to: [cleaned_file_path]")