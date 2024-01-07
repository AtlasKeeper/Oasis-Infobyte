import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

#dataset
df_india = pd.read_csv('A:\\Code\\Projects\\Oasis InfoByte\\Oasis-Infobyte\\unemployment\\Unemployment_India.csv')
df_up_to_11_2020 = pd.read_csv('A:\\Code\\Projects\\Oasis InfoByte\\Oasis-Infobyte\\unemployment\\Unemployment_Rate_upto_11_2020.csv')


#Exploring the dataset
print(df_india.head())