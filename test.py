# dict = {'NORTH': 360, 'NNE': 25, 'NE': 45, 'ENE': 65, 'EAST': 90,
#         'ESE': 115, 'SE': 135, 'SSE': 155, 'SOUTH': 180,
#         'SSW': 205, 'SW': 225, 'WSW': 245, 'WEST': 270,
#         'WNW': 295, 'NW': 315, 'NNW': 335}
#
# dict.radiansdict.get(key, default=None)


arr = [ 1, 2, 3]
s = str( arr)
print(arr)







# import numpy as np
# from scipy import interpolate
# import pylab as pl
# x=np.linspace(0,10,11)
# y=np.sin(x)
# xnew=np.linspace(0,10,101)
# pl. plot(x,y,'ro')
# for kind in ['nearest', 'zero', 'linear','quadratic']:
#        f=interpolate.interp1d(x,y, kind=kind)
#        ynew=f(xnew)
#        pl. plot(xnew, ynew, label=str(kind))
# pl.legend(loc=' lower right')
# pl.show()


# import time
# from datetime import datetime
# a = int(time.time())    #当前时间
# c = datetime.fromtimestamp(a+43200).strftime('%H:%M')    #格式转换
# print(a)
#
# import pandas as pd
# df = {'DataBase':['mysql','test','test','test','test'],'table':['user','student','course','sc','book']}
# df = pd.DataFrame(df)
# # print(df)
# # print(df.iloc[3-1][1])
# #df.drop(index=(df.loc[(df['DataBase']=='test')].index))
# for i in df['table']:
#     if str(i) == 'sc':
#         a = df[df.table==i].index
#         print(a)
#         print(df.iloc[a-1,0])

from itertools import tee, islice, chain, zip_longest

# def previous_and_next(some_iterable):
#     prevs,items, nexts = tee(some_iterable, 3)
#     prevs = chain([None], prevs)
#     nexts = chain(islice(nexts, 1, None), [None])
#     return zip_longest(prevs,items, nexts)
# mylist = ['banana', 'orange', 'apple', 'kiwi', 'tomato']
#
# for item, nxt in previous_and_next(mylist):
#     print("Item is now", item, "next is", nxt)
import datetime
import pandas as pd

# def get_no_date(date_str_li, start_date='', end_date=''):
#     """获取没有列表中没有包含的的日期区间的日期
#     args:
#         start_date: 查询的起始日期字符串，默认为date_li中最小值
#         end_date: 查询的终止日期的字符串, 默认为date_li中最大值
#         date_str_li： 所有需要查询的日期的列表
#     """
#     if not date_str_li:
#         raise ValueError('list can\'t empty')
#     # 所有文件名称，日期的列表
#     try:
#         date_li = [datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') for date_time in date_str_li]
#     except:
#         raise ValueError('your values can\'t  be converted')
#     if end_date:
#         date_end = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
#     else:
#         date_end = max(date_li)
#     if start_date:
#         date_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
#     else:
#         date_start = min(date_li)
#     no_list = []
#     while True:
#         if date_end not in date_li:
#             no_list.append(date_end)
#         if date_end == date_start:
#             break
#         date_end -= datetime.timedelta(hours=1)
#     print([datetime.date.strftime(hour, '%Y-%m-%d %H:%M:%S') for hour in no_list])
#
# lii = []
# df = pd.read_csv('data.csv', usecols=['date'])
# for i in df['date']:
#     lii.append(i)
# #print(lii)
# get_no_date(lii, start_date='2014-01-01 00:51:00', end_date='2016-12-31 23:51:00')


# df = pd.DataFrame({'BoolCol': [1, 2, 3, 3, 4],'attr': [22, 33, 22, 44, 66]},
#        index=[10,20,30,40,50])
# print(df)
# a = df[(df.BoolCol==3)&(df.attr==22)].index.tolist()
# print(a)




