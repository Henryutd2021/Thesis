import pandas as pd


df = pd.read_csv('load_data.csv')

print(df['ERCOT'].isnull().sum())