# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 17:41:38 2022

@author: A109021
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 爬蟲工具
import requests
from bs4 import BeautifulSoup 
import requests
from bs4 import BeautifulSoup

# 資料分析工具
from sklearn.feature_selection import SelectKBest, f_regression

# 正規化工具
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Low Variance Filter
from sklearn.feature_selection import VarianceThreshold

# Feature Importance Filter
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Random forest Regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Linear Regression
from sklearn.linear_model import LinearRegression

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# NN Sk-learn
from sklearn.neural_network import MLPClassifier

# NN Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Cross-validation
from sklearn.model_selection import cross_val_score

# Output
import joblib


#######################    原始資料處理    #######################
# 抓取資料
LIKP = pd.read_csv("./Source/銷售資料/交貨資料(表頭).csv",delimiter=",",header="infer")
LIPS = pd.read_csv("./Source/銷售資料/交貨資料.csv",delimiter=",",header="infer")

# 表頭表身資料合併
merge_data_1 = pd.merge(LIKP, LIPS, how="inner", on="VBELN")
merge_data_1.to_csv('./Process_data/1_menge_likp_lips.csv')


# 刪除數量為 0 項目
merge_data_1.drop(merge_data_1[merge_data_1.MENGE.isnull()].index, inplace=True)

# 刪除幣別缺失項目(加工)
merge_data_1.drop(merge_data_1[merge_data_1.WAERS.isnull()].index, inplace=True)

# 刪除服務性料號
merge_data_1.drop(merge_data_1[merge_data_1.MATNR.str.contains('9989')].index, inplace=True)

# 轉換日期 YYYY/MM/DD => YYYY/MM
merge_data_1['ERDAT'] = pd.to_datetime(merge_data_1['ERDAT'])
merge_data_1['ERDAT'] = merge_data_1['ERDAT'].astype(str).str.replace('-', '').str[0:6]
merge_data_1['ERDAT'] = merge_data_1['ERDAT'].astype(int)


# 依照 物料(MATNR)、建立日期(ERDAT) 排序
merge_data_1 = merge_data_1.sort_values(by=['MATNR','ERDAT'])

# 移除 交貨單 欄位資料
merge_data_1 = merge_data_1.drop(['VBELP'],axis=1)
merge_data_1.to_csv('./Process_data/2_delete_loss_data.csv')



# 合併匯率資料 => 需重新抓取檔案已排除 int64 問題
merge_tmp = pd.read_csv("./Process_data/2_delete_loss_data.csv",delimiter=",",header="infer")
UKURS = pd.read_csv("./Source/銷售資料/匯率.csv",delimiter=",",header="infer")

merge_data_2 = pd.merge(merge_tmp, UKURS, how="inner", on=['WAERS','ERDAT']) 
merge_data_2.to_csv('./Process_data/3_merge_ukurs.csv')



# 依照 物料(MATNR)、建立日期(ERDAT) 進行匯總
sum_menge = merge_data_2.groupby(['MATNR', 'TYPE',  'VKORG', 
                                  'WAERS', 'UKURS', 'MATKL', 
                                  'KUNNR',
                                  'ERDAT', 'UNIT'])['MENGE'].sum()

sum_menge.to_csv('./Process_data/4_menge_sum.csv')

# 合併經濟資料
trading_data = pd.read_csv("./Source/銷售資料/經濟資料.csv",delimiter=",",header="infer")
sum_menge = pd.read_csv('./Process_data/4_menge_sum.csv')
merge_trading = pd.merge(sum_menge, trading_data, how="right", on=['ERDAT'])

merge_trading.to_csv('./Process_data/5_menge_trading.csv')

# 文字資料轉換
le = LabelEncoder()

data = pd.read_csv('./Process_data/5_menge_trading.csv')

data['MATNR'] = le.fit_transform(data['MATNR'])
#print(data['MATNR'].unique())

data['KUNNR'] = le.fit_transform(data['KUNNR'])
#print(data['KUNNR'].unique())

data['TYPE'] = le.fit_transform(data['TYPE'])
#print(data['TYPE'].unique())

data['WAERS'] = le.fit_transform(data['WAERS'])
#print(data['WAERS'].unique())

data['MATKL'] = le.fit_transform(data['MATKL'])
#print(data['MATKL'].unique())

data['UNIT'] = le.fit_transform(data['UNIT'])
#print(data['UNIT'].unique())

data.to_csv('./Process_data/6_data_output.csv')

# 依造銷售組織切分資料

#print(merge_data[ merge_data['VKORG'].astype(str).str.fullmatch('1001')].index)


#######################    資料正規化    #######################

ss = StandardScaler().fit(data)
ss_scaled_data = ss.fit_transform(data)
df = pd.DataFrame(ss_scaled_data)

#print(df.head(10))

#######################     資料分析     #######################
print("資料分析")
X = df.drop([10],axis=1)

Y = df.drop([ 0, 1, 2,
              3, 4, 5,
              6, 7, 8,
              9, 11, 12 ], axis=1)

# Feature Importance 
kb_regr = SelectKBest(f_regression)
X_b = kb_regr.fit_transform(X, Y)
print("資料重要度：" + str(kb_regr.scores_))
print()

#######################    Model 訓練    #######################
X,Y = make_classification(n_samples=64738,
                          n_features=12,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=10, test_size= 0.25)

# Random Forest Regression

RF = RandomForestClassifier(n_estimators=100, random_state=0)

RF_R = RandomForestRegressor(n_estimators=12)
RF_R.fit(X_train, Y_train)

print("Random Forest Regression Score："+str(RF_R.score(X_test, Y_test)))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, Y_train)

print("Linear Regression Score："+str(lr.score(X_test, Y_test)))

# Polynomial Regression
pf = PolynomialFeatures(degree=2)
Xp = pf.fit_transform(X)

lr_p = LinearRegression()
lr_p.fit(Xp, Y)
print('Polynomial regression Score %.3f' % lr_p.score(Xp, Y))
print()

# Neural network  Regression

model = Sequential()
model.add(Dense(24, activation='relu', input_shape=(12,)))
model.add(Dense(12, activation='tanh'))
model.add(Dense(10000, activation='tanh'))
model.add(Dense(1,))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse'])
model.summary()

model.fit(X_train, Y_train, epochs=10)

print()

#######################  Cross-validation  #######################

lr_scores = cross_val_score(lr, X, Y, cv=5)
print("Linear Regression Cross-validation：" + str(lr_scores))

#######################    Metrics 數據    #######################

#######################      Output       #######################

# Sklearn output
joblib.dump(RF_R,'./Output/RandomForest.pkl')
joblib.dump(lr,'./Output/LinearRegression.pkl')
joblib.dump(lr_p,'./Output/Polynomial Regression.pkl')

# Tensorflow
model.save('/Output/')
