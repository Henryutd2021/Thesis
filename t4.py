import pandas as pd
from itertools import tee, islice, chain, zip_longest
import datetime


df = pd.concat(pd.read_excel("data.xlsx", sheet_name=None), ignore_index=True, sort=False)


def previous_and_next(some_iterable):
    prevs, items, nexts = tee(some_iterable, 3)
    nexts = chain(islice(nexts, 1, None), [None])
    prevs = chain([None], prevs)
    return zip_longest(prevs, items, nexts)


def get_no_date(date_str_li, start_date='', end_date=''):

    if not date_str_li:
        raise ValueError('list can\'t empty')
    try:
        date_li = [datetime.datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S') for date_time in date_str_li]
    except:
        raise ValueError('your values can\'t  be converted')
    if end_date:
        date_end = datetime.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    else:
        date_end = max(date_li)
    if start_date:
        date_start = datetime.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    else:
        date_start = min(date_li)
    no_list = []
    while True:
        if date_end not in date_li:
            no_list.append(date_end)
        if date_end == date_start:
            break
        date_end -= datetime.timedelta(hours=1)
    return [datetime.date.strftime(hour, '%Y-%m-%d %H:%M:%S') for hour in no_list]


for prevs, item, nxt in previous_and_next(df['date']):
    if nxt and prevs != None:
        # item = item.to_pydatetime()
        # nxt = nxt.to_pydatetime()
        # prevs = prevs.to_pydatetime()
        if item.minute != 51 and [item.hour == nxt.hour or item.hour == prevs.hour]:
            df.drop(index=(df.loc[(df['date'] == item)].index), inplace=True)


df.drop_duplicates(subset='date', keep="first",inplace=True)
helper = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max(), freq="H")})

df = pd.merge(df, helper, on='date', how='outer').sort_values('date')

print(df.shape)
outputpath='data.csv'
df.to_csv(outputpath,sep=',',index=False,header=True)


lii = []
df = pd.read_csv('data.csv', usecols=['date'])
for i in df['date']:
    lii.append(i)

no_list = get_no_date(lii, start_date='2014-01-01 00:51:00', end_date='2016-12-31 23:51:00')
print(no_list)


