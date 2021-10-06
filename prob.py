import pandas as pd

import numpy as np

from sklearn.metrics import r2_score

df1 = pd.read_csv('vulcancoin-part1.csv')

df2 = pd.read_csv('vulcancoin-part2.csv')

df = pd.concat([df1,df2],ignore_index=True)

df.set_index('date')


# df['date'] = pd.to_datetime(df['date'].apply(lambda x: "-".join(x.replace(',',' ').split())))
#
df = df.sort_index()

df = df.fillna(3)

df['Section_Number'] = df['tx_amount'].str.replace('([A-Z]+)', '')
df['Section_Letter'] = df['tx_amount'].str.extract('([A-Z]+)')

print(r2_score(df['Section_Number'], df['tx_fee']))


# xx = np.corrcoef(df['tx_amount'], df['tx_fee'])
# cir_xx = xx[0, 1]
# r_s = cir_xx**2
#print(r_s)
# df = df.sort_values(by=["date"] , ascending=False)
#
#df.to_csv('1234.csv')
print(df)