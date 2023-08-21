# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 11:46:53 2023

@author: gaoji
"""


import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import numpy as np
import datetime 
from chinese_calendar import is_workday
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# import energy consumption data
sub_sheet_name = ["二层插座","二层照明","一层插座","一层照明","空调"]
data_dict = {}
for name in sub_sheet_name:
    data_dict[name] = pd.read_csv('Data_A/' + name + '_EC.csv')
    data_dict[name]["time"] = pd.to_datetime(data_dict[name]["time"])
    
# import weather info

wh = pd.read_csv("Data_A/天气.csv")
wh["日期"] = pd.to_datetime(wh["日期"])
wh["time"] = wh["日期"] + pd.to_timedelta(wh["小时"],"h")
#wh.drop(["Unnamed:0","日期","小时"], axis=1, inplace=True)

# missing value processing

data = pd.merge(data_dict["二层插座"], data_dict["二层照明"], on = 'time', how = 'inner', suffixes=('_socket_2','_light_2'))
data = pd.merge(data,data_dict["一层插座"],on='time',how='inner',suffixes=('','_socket_1'))
data = pd.merge(data,data_dict["一层照明"],on='time',how='inner',suffixes=('','_light_1'))
data = pd.merge(data,data_dict["空调"],on='time',how='inner',suffixes=('','_air'))
data = data.rename(columns={"value":"value_socket_1"})


#一层二层插座能耗相加，一层二层照明能耗相加。构造数据
data["value_socket"] = data["value_socket_2"] + data["value_socket_1"]
data["value_light"] = data["value_light_2"] + data["value_light_1"]
data.drop(['value_socket_2','value_light_2','value_socket_1','value_light_1'],axis=1,inplace=True)


# 数据尺度调整为每小时，根据赛题要求选定数据范围，构造特征

data = data[(data['time'] >= '2013-8-03 00:00:00') &( data['time'] <= '2015-08-03 00:00:00')]
if_not_workday = []
for dat in data['time']:
    if_not_workday.append(is_workday(dat))
data['workday'] = if_not_workday
data['hour'] = data['time'].dt.hour
data['week'] = data['time'].dt.weekday
data['day']  = data['time'].dt.day
data['month'] = data['time'].dt.month
data['year'] = data['time'].dt.year
data = data.groupby(['year','month','day','week','hour','workday'])[['value_socket','value_light','value_air']].sum().reset_index()


# 处理天气数据，进行merge
wh['hour'] = wh['time'].dt.hour
wh['week'] = wh['time'].dt.weekday
wh['day']  = wh['time'].dt.day
wh['month'] = wh['time'].dt.month
wh['year'] = wh['time'].dt.year
data = pd.merge(data,wh,on=['hour','week','day','month','year'],how='inner')
data = data.rename(columns={"温度":'temp',"湿度":'humidity',"降雨量":'rainfall',
                            "大气压":'atmos',"风向":'wind_direction',
                            "风向角度":'wind_angle',"风速":'wind_speed',"云量":'cloud'})
le = LabelEncoder()
data['wind_direction'] = le.fit_transform(data['wind_direction'])
data['wind_direction'] = data['wind_direction'].astype('category')
data['workday'] = le.fit_transform(data['workday'])
data['work'] = data['workday'].astype('category')


# 多步预测 
data_socket = data.copy()
for i in range(7*24):
    data_socket['value_socket_{}'.format(i)] = data_socket['value_socket'].shift(-i-1)
data_socket.dropna(inplace=True)
targets = [item for item in data_socket.columns if 'value_socket_' in item]
X_socket = data_socket.drop(targets,axis=1)
y_socket = data_socket[targets]




data_light = data.copy()
for i in range(7*24):
    data_light['value_light_{}'.format(i)] = data_light['value_light'].shift(-i-1)
data_light.dropna(inplace=True)

targets = [item for item in data_light.columns if 'value_light_' in item]
X_light = data_light.drop(targets,axis=1)
y_light = data_light[targets]


data_air = data.copy()
for i in range(7*24):
    data_air['value_air_{}'.format(i)] = data_air['value_air'].shift(-i-1)
data_air.dropna(inplace=True)

targets = [item for item in data_air.columns if 'value_air_' in item]
X_air = data_air.drop(targets,axis=1)
y_air = data_air[targets]


# 保存训练数据，其中输入数据含有  'Unnamed: 0','日期','time','work'
targets_drop = ['Unnamed: 0','日期','time','work']
X_air = X_air.drop(targets_drop,axis=1)
X_light = X_light.drop(targets_drop,axis=1)
X_socket = X_socket.drop(targets_drop,axis=1)
X_air.to_csv('.\\Data_A\\X_air.csv')
X_light.to_csv('.\\Data_A\\X_light.csv')
X_socket.to_csv('.\\Data_A\\X_socket.csv')
y_air.to_csv('.\\Data_A\\y_air.csv')
y_light.to_csv('.\\Data_A\\y_light.csv')
y_socket.to_csv('.\\Data_A\\y_socket.csv')

