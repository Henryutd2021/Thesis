import pandas as pd
from itertools import tee, islice, chain, zip_longest
import numpy as np
import datetime
import math


df = pd.concat(pd.read_excel("data.xlsx", sheet_name=None), ignore_index=True, sort=False)
helper = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max(), freq="H")})
df = pd.merge(df, helper, on='date', how='outer').sort_values('date')


def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    nexts = chain(islice(nexts, 1, None), [None])
    prevs = chain([None], prevs)
    return zip_longest(prevs, items, nexts)


# for prevs, item, nxt in previous_and_next(df['date']):
#     if nxt and prevs != None:
#         if math.fabs(item.minute - 51) > 5 and [item.hour == nxt.hour or item.hour == prevs.hour]:
#             df.drop(index=(df.loc[(df['date'] == item)].index), inplace=True)


df.drop_duplicates(subset='date', keep="first",inplace=True)
#df.fillna(method='ffill', limit=5, inplace=True)

for prevs, item, nxt in previous_and_next(df['temp (F)']):
    if pd.isna(item):
        if pd.isna(prevs):
            pass
        elif pd.isna(nxt):
            item = prevs
        else:
            item = np.mean([prevs, nxt])
            a = df[(df.temp (F) == item) & (df.attr == item)].index.tolist()
            print(a)






print(df.shape)
df.to_csv('data.csv',sep=',',index=False,header=True)
