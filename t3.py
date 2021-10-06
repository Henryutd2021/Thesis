import pandas as pd
import numpy as np


df = pd.concat(pd.read_excel("data.xlsx", sheet_name=None), ignore_index=True, sort=False)
helper = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max(), freq="H")})
df = pd.merge(df, helper, on='date', how='outer').sort_values('date')
df.drop_duplicates(subset='date', keep="last",inplace=True)

df['temp'] = df['temp'].interpolate(method='linear')
df.to_csv('data.csv',sep=',',index=False,header=True)
# for i in list(np.where(pd.isna(df['temp']))):
#     for j in i:
#     #     print(j)
#
#         print(df.iloc[j, 1])
# list = []
# for j in list(np.where(pd.isna(df['temp']))):
#     for i in df['temp'].isna():
#         if i == False:
#             if len(list) == 0:
#                 list.append(df.iloc[j, 1])
#             elif len(list) == 1:
#                 list[0] = df.iloc[j, 1]
#             else:
#                 list.append(df.iloc[j, 1])
#                 print(list)
#                 list = []
#         else:
#             if len(list) == 0:
#                 pass
#             else:
#                 list.append(df.iloc[j, 1])









#df.set_index('date', inplace=True)
#print((pd.isna(df).find()))

#df = np.array(df)
# for i in range(len(df['temp'])):
#     if np.isnan(df.loc['temp', i]) == True:
#         print(i)
    #print(np.where(np.isnan(df['temp'][i])))
#print(np.where(pd.isna(df['temp'])))
# for i in df['temp']:
#     if str(i) == 'nan':
#         print(i.index())
        #print(np.where(i))
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
    if i == False:
        if len(list) == 0:
            list.append(i)
        elif len(list) == 1:
            list[0] = i
        else:
            list.append(i)
            print(list)
            list = []
    else:
        if len(list) == 0:
            pass
        else:
            list.append(i)



# def get_data(x, x1, y1, x2, y2):
#     y = x*(y2-y1)/(x2-x1)
#     return y

