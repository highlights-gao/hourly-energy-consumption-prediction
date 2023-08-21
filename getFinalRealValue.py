# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:33:22 2023

@author: gaoji
"""

import pandas as pd


data_ = ['一层插座','一层照明','二层插座','二层照明','空调']

data_dict = {}
for data in data_:
    path = '.\\Data_B\\' + data + '_EC_B.csv'
    data_dict[data] = pd.read_csv(path)
    
  
data = pd.merge(data_dict["二层插座"], data_dict["二层照明"], on = 'time', how = 'inner', suffixes=('_socket_2','_light_2'))
data = pd.merge(data,data_dict["一层插座"],on='time',how='inner',suffixes=('','_socket_1'))
data = pd.merge(data,data_dict["一层照明"],on='time',how='inner',suffixes=('','_light_1'))
data = pd.merge(data,data_dict["空调"],on='time',how='inner',suffixes=('','_air'))
data = data.rename(columns={"value":"value_socket_1"})

#一层二层插座能耗相加，一层二层照明能耗相加。构造数据
data["value_socket"] = data["value_socket_2"] + data["value_socket_1"]
data["value_light"] = data["value_light_2"] + data["value_light_1"]
data.drop(['value_socket_2','value_light_2','value_socket_1','value_light_1'],axis=1,inplace=True)
data['total'] = data['value_socket'] + data['value_light'] + data['value_air']


data.to_csv('.\\Data_B\\最后四周真实数据.csv')