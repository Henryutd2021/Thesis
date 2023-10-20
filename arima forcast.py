import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


data = pd.read_csv('load_data_with weather without nan.csv')
data['Hour_End'] = pd.to_datetime(data['Hour_End'])
data.set_index(['Hour_End'], inplace=True)
#print(data.head())

# print(data.isnull().values.any())
#
# plt.figure(figsize=(20,4))
# ax = plt.gca()
# ax.set_xlabel('Date')
# ax.set_ylabel('Mwh')
# plt.plot(data.index, data['ERCOT'])
# #plt.plot(data['Hour_End'], data['ERCOT'])
# plt.show()


def test_stationarity(ts):
    stats = ['Test Statistic','p-value','Lags','Observations']
    df_test = adfuller(ts, autolag='AIC')
    df_results = pd.Series(df_test[:4], index=stats)
    for key,value in df_test[4].items():
        df_results[f'Critical Value ({key})'] = value
    print(df_results)


#test_stationarity(data['ERCOT'])
data['diff'] = data['ERCOT'] - data['ERCOT'].shift(1)
test_stationarity(data['diff'].dropna(inplace=False))


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data['diff'][1:], lags=40, ax=ax1)   # first value of diff is NaN
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data['diff'][1:], lags=40, ax=ax2)
#plt.show()


data['ERCOT.float'] = data['ERCOT'].astype(float)


fig = plt.figure(figsize=(20,8))
model = ARIMA(data['ERCOT.float'], order=(1,0,0))
ax = plt.gca()
results = model.fit()
plt.plot(data['ERCOT'])
plt.plot(results.fittedvalues, color='red')
ax.legend(['ERCOT', 'Forecast'])
# plt.show()
# print(results.summary())

# fig = plt.figure(figsize=(20,8))
# model = ARIMA(data['ERCOT.float'], order=(1,0,1))
# ax = plt.gca()
# results = model.fit()
# plt.plot(data['ERCOT'])
# plt.plot(results.fittedvalues, color='red')
# ax.legend(['ERCOT', 'Forecast'])

# plt.show()
# print(results.summary())


# fig = plt.figure(figsize=(20,8))
# model = ARIMA(data['ERCOT'], order=(1,0,0))
# ax = plt.gca()
# results = model.fit()
# plt.plot(data['ERCOT'])
# plt.plot(results.fittedvalues, color='red')
# ax.legend(['Car Count', 'Forecast'])
#
# print results.summary()


# forecast quick and dirty

fig = plt.figure(figsize=(20,8))
num_points = len(data['ERCOT'])
x = results.predict(start=(300), end=(352), dynamic=False)

plt.plot(data['ERCOT'][250:300])
plt.plot(data['ERCOT'][300:352],color='b')
plt.plot(x, color='r')
print(results.summary())
plt.show()
