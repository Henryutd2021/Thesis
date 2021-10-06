import pandas as pd
import numpy as np
import datetime

starttime = datetime.datetime.now()
whether = {
    'BLOWING SNOW': 1,
    'CLEAR': 10,
    'FOG': 4,
    'HAZE': 3,
    'HEAVY RAIN': 0,
    'HEAVY SNOW': 0,
    'HEAVY THUNDERSTORMS AND RAIN': 0,
    'LIGHT DRIZZLE': 2,
    'LIGHT FREEZING DRIZZLE': 2,
    'LIGHT FREEZING RAIN': 2,
    'LIGHT ICE PELLETS': 2,
    'LIGHT RAIN': 2,
    'LIGHT SNOW': 2,
    'LIGHT THUNDERSTORMS AND RAIN': 1,
    'MIST': 5,
    'MOSTLY CLOUDY': 6,
    'OVERCAST': 5,
    'PARTLY CLOUDY': 8,
    'PATCHES OF FOG': 5,
    'RAIN': 1,
    'SCATTERED CLOUDS': 7,
    'SHALLOW FOG': 5,
    'SNOW': 1,
    'THUNDERSTORM': 0,
    'THUNDERSTORM ADN RAIN': 0,
    'UNKNOWN': np.nan}

wind = {'NORTH': 360, 'NNE': 25, 'NE': 45, 'ENE': 65, 'EAST': 90,
        'ESE': 115, 'SE': 135, 'SSE': 155, 'SOUTH': 180,
        'SSW': 205, 'SW': 225, 'WSW': 245, 'WEST': 270,
        'WNW': 295, 'NW': 315, 'NNW': 335}

df = pd.concat(pd.read_excel("New York.xlsx", sheet_name=None), ignore_index=True, sort=False)
df.drop(
    labels=['temp', 'wind_spd', 'wind_gust', 'wind_chill', 'heat_index', 'precip', 'precip_rate', 'precip_total', 'fog',
            'rain', 'snow', 'hail', 'thunder', 'tornado', 'Condition'], axis=1, inplace=True)
helper = pd.DataFrame({'date': pd.date_range(df['date'].min(), df['date'].max(), freq="H")})
df = pd.merge(df, helper, on='date', how='outer').sort_values('date')
df.drop_duplicates(subset='date', keep="last", inplace=True)

for key in df['dir']:
    df['dir'].replace(str(key), wind.get(str(key).upper(), np.nan), inplace=True)

for key in df['cond']:
    df['cond'].replace(str(key), whether.get(str(key).upper(), np.nan), inplace=True)

list = ['Temp (C )', 'dew_pt', 'hum', 'Wind Speed (m/s)', 'vis', 'pressure', 'dir', 'cond']
for i in list:
    df[i] = df[i].interpolate(method='linear')
for item in df['date']:
    if item.minute != 51:
        df.drop(index=df.loc[(df['date'] == item)].index, inplace=True)

df.to_csv('data.csv', sep=',', index=False, header=True)
print(df.shape)
endtime = datetime.datetime.now()
print('Time: ', endtime - starttime)
