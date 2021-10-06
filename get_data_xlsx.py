import os
import pandas as pd
from pandas import to_datetime


files = os.listdir('./load data')

df = pd.DataFrame()

for f in files:
    data = pd.read_excel('./load data/'+ f, sort=True)
    df = df.append(data)

#df['Hour_End'] = pd.to_datetime(df['Hour_End'], format='%Y/%m/%d %H:%M')
#df = to_datetime(df.Hour_End, format="%Y/%m/%d %H")
df.to_csv('load_data.csv', sep=',', index=False, header=True)
print(df.head())






















