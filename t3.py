import pandas as pd
import numpy as np


df = pd.concat(pd.read_excel("data.xlsx", sheet_name=None), ignore_index=True, sort=False)
helper = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max(), freq="H")})
df = pd.merge(df, helper, on='date', how='outer').sort_values('date')
df.drop_duplicates(subset='date', keep="last",inplace=True)

df['temp'] = df['temp'].interpolate(method='linear')
df.to_csv('data.csv',sep=',',index=False,header=True)
#print([i[2] for i in df['temp']])

#     if pd.isna(i) == True:
#         print(df.iloc[int(np.where(i)), 1])

#print(df)
# list = []
# for i in df['temp']:
#     if i == Nan:
#         if len(list) == 0:
#             list.append(i)
#         elif len(list) == 1:
#             list[0] = i
#         else:
#             list.append(i)
#             print(list)
#             list = []
#     else:
#         if len(list) == 0:
#             pass
#         else:
#             list.append(i)


list = []
for i in df['temp'].isna():
    if i == False and len(list) == 0 or i != False and len(list) != 0:
        list.append(i)
    elif i == False and len(list) == 1:
        list[0] = i
    elif i == False:
        list.append(i)
        print(list)
        list = []



# def get_data(x, x1, y1, x2, y2):
#     y = x*(y2-y1)/(x2-x1)
#     return y

