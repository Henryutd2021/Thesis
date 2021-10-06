# coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = pd.read_excel('指数.xlsx',header=None,index_col=None)

# 数据信息
# print(data.info())

# 查看空值
isnull = data[1].isnull()
# print(isnull)
# print(data[1])

# 替换空值
data[1] = data[1].fillna('666')

# 找出索引
index_ = data[isnull].index.tolist()
# print(index_)

# 去除空列所在行
data = data.drop(index_)
# print(data)

x = data[1]
y = data[0]

# 插值

f1=interp1d(x,y,kind='linear')#线性插值
f2=interp1d(x,y,kind='cubic')#三次样条插值
x_pred=np.arange(1,170,1)
y1=f1(x_pred)

datas = pd.DataFrame([y1,x_pred])
datas.to_excel('new指数.xlsx')


y2=f2(x_pred)
plt.figure(figsize=[12,7])
plt.scatter(x,y,s=30,c='red',label='原始指数')
plt.plot(x_pred,y1,'b--',label='linear interpolation')
# plt.plot(x_pred,y2,'b--',label='cubic')
plt.legend(loc='upper left')
font_size = {'size':13}
plt.ylabel('淘宝指数',font_size)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.show()