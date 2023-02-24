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

# 資料分析工具
from sklearn.feature_selection import VarianceThreshold,SelectKBest, f_regression

# 正規化工具
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Low Variance Filter
from sklearn.feature_selection import VarianceThreshold

# Feature Importance Filter
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Random forest Regression
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

# Metrics
from sklearn import metrics

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

data = pd.read_csv('./Process_data/5_menge_trading.csv')

data = data.drop(['WAERS'], axis=1)
data.to_csv('./Process_data/6_data_output.csv')

# 依造銷售組織切分資料
data_1001 = data.query('(VKORG==1001) or (VKORG==1004)')
data_1001.to_csv('./Process_data/7_data_1001.csv')

data_1002 = data.query('(VKORG==1002) or (VKORG==1011) or (VKORG==2002)')
data_1002.to_csv('./Process_data/7_data_1002.csv')

data_1003 = data.query('VKORG==1003')
data_1003.to_csv('./Process_data/7_data_1003.csv')

data_1005 = data.query('(VKORG==1005) or (VKORG==1007)')
data_1005.to_csv('./Process_data/7_data_1005.csv')

data_1006 = data.query('VKORG==1006')
data_1006.to_csv('./Process_data/7_data_1006.csv')

# 文字資料轉換
le = LabelEncoder()

data['MATNR'] = le.fit_transform(data['MATNR'])
data['KUNNR'] = le.fit_transform(data['KUNNR'])
data['TYPE'] = le.fit_transform(data['TYPE'])
data['MATKL'] = le.fit_transform(data['MATKL'])
data['UNIT'] = le.fit_transform(data['UNIT'])

data_1001['MATNR'] = le.fit_transform(data_1001['MATNR'])
data_1001['KUNNR'] = le.fit_transform(data_1001['KUNNR'])
data_1001['TYPE'] = le.fit_transform(data_1001['TYPE'])
data_1001['MATKL'] = le.fit_transform(data_1001['MATKL'])
data_1001['UNIT'] = le.fit_transform(data_1001['UNIT'])

data_1002['MATNR'] = le.fit_transform(data_1002['MATNR'])
data_1002['KUNNR'] = le.fit_transform(data_1002['KUNNR'])
data_1002['TYPE'] = le.fit_transform(data_1002['TYPE'])
data_1002['MATKL'] = le.fit_transform(data_1002['MATKL'])
data_1002['UNIT'] = le.fit_transform(data_1002['UNIT'])

data_1003['MATNR'] = le.fit_transform(data_1003['MATNR'])
data_1003['KUNNR'] = le.fit_transform(data_1003['KUNNR'])
data_1003['TYPE'] = le.fit_transform(data_1003['TYPE'])
data_1003['MATKL'] = le.fit_transform(data_1003['MATKL'])
data_1003['UNIT'] = le.fit_transform(data_1003['UNIT'])

data_1005['MATNR'] = le.fit_transform(data_1005['MATNR'])
data_1005['KUNNR'] = le.fit_transform(data_1005['KUNNR'])
data_1005['TYPE'] = le.fit_transform(data_1005['TYPE'])
data_1005['MATKL'] = le.fit_transform(data_1005['MATKL'])
data_1005['UNIT'] = le.fit_transform(data_1005['UNIT'])

data_1006['MATNR'] = le.fit_transform(data_1006['MATNR'])
data_1006['KUNNR'] = le.fit_transform(data_1006['KUNNR'])
data_1006['TYPE'] = le.fit_transform(data_1006['TYPE'])
data_1006['MATKL'] = le.fit_transform(data_1006['MATKL'])
data_1006['UNIT'] = le.fit_transform(data_1006['UNIT'])


#######################    資料標準化    #######################
# 使用 Standard
col_names = ['UKURS','MENGE','YOY','EAGRT']

scaled_features = data.copy()
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features[col_names] = features
df = pd.DataFrame(scaled_features)

scaled_features_1001 = data_1001.copy()
features_1001 = scaled_features_1001[col_names]
scaler_1001 = StandardScaler().fit(features_1001.values)
features_1001 = scaler.transform(features_1001.values)
scaled_features_1001[col_names] = features_1001
df_1001 = pd.DataFrame(scaled_features_1001)

scaled_features_1002 = data_1002.copy()
features_1002 = scaled_features_1002[col_names]
scaler_1002 = StandardScaler().fit(features_1002.values)
features_1002 = scaler.transform(features_1002.values)
scaled_features_1002[col_names] = features_1002
df_1002 = pd.DataFrame(scaled_features_1002)

scaled_features_1003 = data_1003.copy()
features_1003 = scaled_features_1003[col_names]
scaler_1003 = StandardScaler().fit(features_1003.values)
features_1003 = scaler.transform(features_1003.values)
scaled_features_1003[col_names] = features_1003
df_1003 = pd.DataFrame(scaled_features_1003)

scaled_features_1005 = data_1005.copy()
features_1005 = scaled_features_1005[col_names]
scaler_1005 = StandardScaler().fit(features_1005.values)
features_1005 = scaler.transform(features_1005.values)
scaled_features_1005[col_names] = features_1005
df_1005 = pd.DataFrame(scaled_features_1005)

scaled_features_1006 = data_1006.copy()
features_1006 = scaled_features_1006[col_names]
scaler_1006 = StandardScaler().fit(features_1006.values)
features_1006 = scaler.transform(features_1006.values)
scaled_features_1006[col_names] = features_1006
df_1006 = pd.DataFrame(scaled_features_1006)

#print(df.head(10))

#######################     資料分析     #######################
print("資料分析")

# 關聯性分析
df.corr().to_csv('./Process_data/8_data_relation.csv')
df_1001.corr().to_csv('./Process_data/8_relation_1001.csv')
df_1002.corr().to_csv('./Process_data/8_relation_1002.csv')
df_1003.corr().to_csv('./Process_data/8_relation_1003.csv')
df_1005.corr().to_csv('./Process_data/8_relation_1005.csv')
df_1006.corr().to_csv('./Process_data/8_relation_1006.csv')

print("1001 資料關聯性分析：" + str(df_1001.corr()))
print("1002 資料關聯性分析：" + str(df_1002.corr()))
print("1003 資料關聯性分析：" + str(df_1003.corr()))
print("1005 資料關聯性分析：" + str(df_1005.corr()))
print("1006 資料關聯性分析：" + str(df_1006.corr()))
print()

X = df.drop([ 'MENGE', 'Unnamed: 0' ],axis=1)
Y = df.drop([ 'MATNR',	'TYPE',	 'VKORG',	
              'UKURS',  'MATKL', 'KUNNR',	
              'ERDAT',  'UNIT',  'YOY'  ,
              'EAGRT', 'Unnamed: 0' ], axis=1)

X_1001 = df_1001.drop([ 'MENGE', 'Unnamed: 0' ], axis=1)
Y_1001 = df_1001.drop([ 'MATNR',  'TYPE',  'VKORG',	
                        'UKURS',  'MATKL', 'KUNNR',	
                        'ERDAT',  'UNIT',  'YOY'  ,
                        'EAGRT',  'Unnamed: 0' ], axis=1)


X_1002 = df_1002.drop([ 'MENGE', 'Unnamed: 0'], axis=1)
Y_1002 = df_1002.drop([ 'MATNR',  'TYPE',  'VKORG',	
                        'UKURS',  'MATKL', 'KUNNR',	
                        'ERDAT',  'UNIT',  'YOY'  ,
                        'EAGRT',  'Unnamed: 0' ], axis=1)


X_1003 = df_1003.drop([ 'MENGE', 'Unnamed: 0' ],axis=1)
Y_1003 = df_1003.drop([ 'MATNR',  'TYPE',  'VKORG',	
                        'UKURS',  'MATKL', 'KUNNR',	
                        'ERDAT',  'UNIT',  'YOY'  ,
                        'EAGRT',  'Unnamed: 0' ], axis=1)


X_1005 = df_1005.drop([ 'MENGE', 'Unnamed: 0' ],axis=1)
Y_1005 = df_1005.drop([ 'MATNR',  'TYPE',  'VKORG',	
                        'UKURS',  'MATKL', 'KUNNR',	
                        'ERDAT',  'UNIT',  'YOY'  ,
                        'EAGRT',  'Unnamed: 0' ], axis=1)


X_1006 = df_1006.drop([ 'MENGE', 'Unnamed: 0' ],axis=1)
Y_1006 = df_1006.drop([ 'MATNR',  'TYPE',  'VKORG',	
                        'UKURS',  'MATKL', 'KUNNR',	
                        'ERDAT',  'UNIT',  'YOY'  ,
                        'EAGRT',  'Unnamed: 0' ], axis=1)

print("資料 Variance 分析： 排除 Variance = 0") 

print("整體資料 Variance")
print(str(X.var(axis=0)))
vt = VarianceThreshold(threshold=0.0)
X = vt.fit_transform(X)
#print("調整後分散度"+ str(X.var(axis=0)))
print()

print("1001 Variance - ")
print(str(X_1001.var(axis=0)))
vt_1001 = VarianceThreshold(threshold=0.0)
X_1001 = vt_1001.fit_transform(X_1001)
#print("1001 調整後分散度"+ str(X_1001.var(axis=0)))
print()

print("1002 Variance - ")
print(str(X_1002.var(axis=0)))
vt_1002 = VarianceThreshold(threshold=0.0)
X_1002 = vt_1002.fit_transform(X_1002)
#print("1002 調整後分散度"+ str(X_1002.var(axis=0)))
print()

print("1003 Variance - ")
print(str(X_1003.var(axis=0)))
vt_1003 = VarianceThreshold(threshold=0.0)
X_1003 = vt_1003.fit_transform(X_1003)
#print("1003 調整後分散度"+ str(X_1003.var(axis=0)))
print()

print("1005 Variance - ")
print(str(X_1005.var(axis=0)))
vt_1005 = VarianceThreshold(threshold=0.0)
X_1005 = vt_1005.fit_transform(X_1005)
#print("1005 調整後分散度"+ str(X_1005.var(axis=0)))
print()

print("1006 Variance - ")
print(str(X_1006.var(axis=0)))
vt_1006 = VarianceThreshold(threshold=0.0)
X_1006 = vt_1006.fit_transform(X_1006)
#print("1006 調整後分散度"+ str(X_1006.var(axis=0)))
print()

# Feature Importance 
kb_regr_1001 = SelectKBest(f_regression)
X_b_1001 = kb_regr_1001.fit_transform(X_1001, Y_1001)
print("1001 資料重要度：" + str(kb_regr_1001.scores_))
print()

kb_regr_1002 = SelectKBest(f_regression)
X_b_1002 = kb_regr_1002.fit_transform(X_1002, Y_1002)
print("1002 資料重要度：" + str(kb_regr_1002.scores_))
print()

kb_regr_1003 = SelectKBest(f_regression,  k="all")
X_b_1003 = kb_regr_1003.fit_transform(X_1003, Y_1003)
print("1003 資料重要度：" + str(kb_regr_1003.scores_))
print()

kb_regr_1005 = SelectKBest(f_regression)
X_b_1005 = kb_regr_1005.fit_transform(X_1005, Y_1005)
print("1005 資料重要度：" + str(kb_regr_1005.scores_))
print()

kb_regr_1006 = SelectKBest(f_regression, k="all")
X_b_1006 = kb_regr_1006.fit_transform(X_1006, Y_1006)
print("1006 資料重要度：" + str(kb_regr_1006.scores_))
print()


#######################    Model 訓練    #######################
print("Model 訓練")
print()

X,Y = make_classification(n_samples=df.shape[0],
                          n_features=11,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_1001,Y_1001 = make_classification(n_samples=df_1001.shape[0],
                          n_features=11,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_1002,Y_1002 = make_classification(n_samples=df_1002.shape[0],
                          n_features=11,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_1003,Y_1003 = make_classification(n_samples=df_1003.shape[0],
                          n_features=11,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_1005,Y_1005 = make_classification(n_samples=df_1005.shape[0],
                          n_features=11,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_1001_train, X_1001_test, Y_1001_train, Y_1001_test = train_test_split(X_1001, Y_1001, random_state=10, test_size= 0.25)
X_1002_train, X_1002_test, Y_1002_train, Y_1002_test = train_test_split(X_1002, Y_1002, random_state=10, test_size= 0.25)
X_1003_train, X_1003_test, Y_1003_train, Y_1003_test = train_test_split(X_1003, Y_1003, random_state=10, test_size= 0.25)
X_1005_train, X_1005_test, Y_1005_train, Y_1005_test = train_test_split(X_1005, Y_1005, random_state=10, test_size= 0.25)
print()

# Random Forest Regression

RF_R_1001 = RandomForestRegressor(n_estimators=11)
RF_R_1001.fit(X_1001_train, Y_1001_train)

RF_R_1002 = RandomForestRegressor(n_estimators=11)
RF_R_1002.fit(X_1002_train, Y_1002_train)

RF_R_1003 = RandomForestRegressor(n_estimators=11)
RF_R_1003.fit(X_1003_train, Y_1003_train)

RF_R_1005 = RandomForestRegressor(n_estimators=11)
RF_R_1005.fit(X_1005_train, Y_1005_train)

print("Random Forest Regression Score：")
print('1001 - ' + str(RF_R_1001.score(X_1001_test, Y_1001_test)))
print('1002 - ' + str(RF_R_1002.score(X_1002_test, Y_1002_test)))
print('1003 - ' + str(RF_R_1003.score(X_1003_test, Y_1003_test)))
print('1005 - ' + str(RF_R_1005.score(X_1005_test, Y_1005_test)))
print()

# Linear Regression
lr_1001 = LinearRegression()
lr_1001.fit(X_1001_train, Y_1001_train)

lr_1002 = LinearRegression()
lr_1002.fit(X_1002_train, Y_1002_train)

lr_1003 = LinearRegression()
lr_1003.fit(X_1003_train, Y_1003_train)

lr_1005 = LinearRegression()
lr_1005.fit(X_1005_train, Y_1005_train)

print("Linear Regression Score：")
print('1001 - ' + str(lr_1001.score(X_1001_test, Y_1001_test)))
print('1002 - ' + str(lr_1002.score(X_1002_test, Y_1002_test)))
print('1003 - ' + str(lr_1003.score(X_1003_test, Y_1003_test)))
print('1005 - ' + str(lr_1005.score(X_1005_test, Y_1005_test)))
print()

# Polynomial Regression
pf_1001 = PolynomialFeatures(degree=3)
Xp_1001 = pf_1001.fit_transform(X_1001)
lr_p_1001 = LinearRegression()
lr_p_1001.fit(Xp_1001, Y_1001)

pf_1002 = PolynomialFeatures(degree=2)
Xp_1002 = pf_1002.fit_transform(X_1002)
lr_p_1002 = LinearRegression()
lr_p_1002.fit(Xp_1002, Y_1002)

pf_1003 = PolynomialFeatures(degree=2)
Xp_1003 = pf_1001.fit_transform(X_1003)
lr_p_1003 = LinearRegression()
lr_p_1003.fit(Xp_1003, Y_1003)

pf_1005 = PolynomialFeatures(degree=2)
Xp_1005 = pf_1005.fit_transform(X_1005)
lr_p_1005 = LinearRegression()
lr_p_1005.fit(Xp_1005, Y_1005)

print('Polynomial regression Score：')
print('1001 - ' + str(lr_p_1001.score(Xp_1001, Y_1001)))
print('1002 - ' + str(lr_p_1002.score(Xp_1002, Y_1002)))
print('1003 - ' + str(lr_p_1003.score(Xp_1003, Y_1003)))
print('1005 - ' + str(lr_p_1005.score(Xp_1005, Y_1005)))
print()

# Neural network Regression
print('Data - Neural network')
model = Sequential()
model.add(Dense(22, activation='relu', input_shape=(11,)))
model.add(Dense(12, activation='tanh'))
model.add(Dense(10000, activation='tanh'))
model.add(Dense(1,))
model.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse','mae'])
model.summary()
print()
#model.fit(X, Y, epochs=10)
print()


print('1001 - Neural network')
model_1001 = Sequential()
model_1001.add(Dense(22, activation='relu', input_shape=(11,)))
model_1001.add(Dense(12, activation='tanh'))
model_1001.add(Dense(10000, activation='tanh'))
model_1001.add(Dense(1,))
model_1001.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse','mae'])
model_1001.fit(X_1001_train, Y_1001_train, epochs=10)
print()

print('1002 - Neural network')
model_1002 = Sequential()
model_1002.add(Dense(22, activation='relu', input_shape=(11,)))
model_1002.add(Dense(12, activation='tanh'))
model_1002.add(Dense(10000, activation='tanh'))
model_1002.add(Dense(1,))
model_1002.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse','mae'])
model_1002.fit(X_1002_train, Y_1002_train, epochs=10)
print()

print('1003 - Neural network')
model_1003 = Sequential()
model_1003.add(Dense(22, activation='relu', input_shape=(11,)))
model_1003.add(Dense(12, activation='tanh'))
model_1003.add(Dense(10000, activation='tanh'))
model_1003.add(Dense(1,))
model_1003.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse','mae'])
model_1003.fit(X_1003_train, Y_1003_train, epochs=10)
print()

print('1005 - Neural network')
model_1005 = Sequential()
model_1005.add(Dense(22, activation='relu', input_shape=(11,)))
model_1005.add(Dense(12, activation='tanh'))
model_1005.add(Dense(10000, activation='tanh'))
model_1005.add(Dense(1,))
model_1005.compile(optimizer='sgd', loss='mse', metrics=['accuracy','mse','mae'])
model_1005.fit(X_1005_train, Y_1005_train, epochs=10)
print()

#######################  Cross-validation  #######################

lr_scores_1001 = cross_val_score(lr_1001, X_1001, Y_1001, cv=5)
lr_scores_1002 = cross_val_score(lr_1002, X_1002, Y_1002, cv=5)
lr_scores_1003 = cross_val_score(lr_1003, X_1003, Y_1003, cv=5)
lr_scores_1005 = cross_val_score(lr_1005, X_1005, Y_1005, cv=5)

print("Linear Regression Cross-validation：")
print('1001 - ' + str(lr_scores_1001))
print('1002 - ' + str(lr_scores_1002))
print('1003 - ' + str(lr_scores_1003))
print('1005 - ' + str(lr_scores_1005))
print()

lr_p_scores_1001 = cross_val_score(lr_p_1001, Xp_1001, Y_1001, cv=5)
lr_p_scores_1002 = cross_val_score(lr_p_1002, Xp_1002, Y_1002, cv=5)
lr_p_scores_1003 = cross_val_score(lr_p_1003, Xp_1003, Y_1003, cv=5)
lr_p_scores_1005 = cross_val_score(lr_p_1005, Xp_1005, Y_1005, cv=5)

print("Polynomial regression Cross-validation：")
print('1001 - ' + str(lr_p_scores_1001))
print('1002 - ' + str(lr_p_scores_1002))
print('1003 - ' + str(lr_p_scores_1003))
print('1005 - ' + str(lr_p_scores_1005))
print()

#######################      Metrics      #######################

print('Random Forest Regression Metrics：')
RF_1001_predicted = RF_R_1001.predict(X_1001_test)
RF_1001_mae = metrics.mean_absolute_error(Y_1001_test, RF_1001_predicted)
RF_1001_mse = metrics.mean_squared_error(Y_1001_test, RF_1001_predicted)
print('1001 Random Forest Regression MAE - ' + str(RF_1001_mae))
print('1001 Random Forest Regression MSE - ' + str(RF_1001_mse))
print()

RF_1002_predicted = RF_R_1002.predict(X_1002_test)
RF_1002_mae = metrics.mean_absolute_error(Y_1002_test, RF_1002_predicted)
RF_1002_mse = metrics.mean_squared_error(Y_1002_test, RF_1002_predicted)
print('1002 Random Forest Regression MAE - ' + str(RF_1002_mae))
print('1002 Random Forest Regression MSE - ' + str(RF_1002_mse))
print()

RF_1003_predicted = RF_R_1003.predict(X_1003_test)
RF_1003_mae = metrics.mean_absolute_error(Y_1003_test, RF_1003_predicted)
RF_1003_mse = metrics.mean_squared_error(Y_1003_test, RF_1003_predicted)
print('1003 Random Forest Regression MAE - ' + str(RF_1003_mae))
print('1003 Random Forest Regression MSE - ' + str(RF_1003_mse))
print()

RF_1005_predicted = RF_R_1005.predict(X_1005_test)
RF_1005_mae = metrics.mean_absolute_error(Y_1005_test, RF_1005_predicted)
RF_1005_mse = metrics.mean_squared_error(Y_1005_test, RF_1005_predicted)
print('1005 Random Forest Regression MAE - ' + str(RF_1005_mae))
print('1005 Random Forest Regression MSE - ' + str(RF_1005_mse))
print()

print('Linear Regression Metrics：')
lr_y_1001_predicted = lr_1001.predict(X_1001_test)
lr_1001_mae = metrics.mean_absolute_error(Y_1001_test, lr_y_1001_predicted)
lr_1001_mse = metrics.mean_squared_error(Y_1001_test, lr_y_1001_predicted)
print('1001 Linear Regression MAE - ' + str(lr_1001_mae))
print('1001 Linear Regression MSE - ' + str(lr_1001_mse))
print()

lr_y_1002_predicted = lr_1002.predict(X_1002_test)
lr_1002_mae = metrics.mean_absolute_error(Y_1002_test, lr_y_1002_predicted)
lr_1002_mse = metrics.mean_squared_error(Y_1002_test, lr_y_1002_predicted)
print('1002 Linear Regression MAE - ' + str(lr_1002_mae))
print('1002 Linear Regression MSE - ' + str(lr_1002_mse))
print()

lr_y_1003_predicted = lr_1003.predict(X_1003_test)
lr_1003_mae = metrics.mean_absolute_error(Y_1003_test, lr_y_1003_predicted)
lr_1003_mse = metrics.mean_squared_error(Y_1003_test, lr_y_1003_predicted)
print('1003 Linear Regression MAE - ' + str(lr_1003_mae))
print('1003 Linear Regression MSE - ' + str(lr_1003_mse))
print()

lr_y_1005_predicted = lr_1005.predict(X_1005_test)
lr_1005_mae = metrics.mean_absolute_error(Y_1005_test, lr_y_1005_predicted)
lr_1005_mse = metrics.mean_squared_error(Y_1005_test, lr_y_1005_predicted)
print('1005 Linear Regression MAE - ' + str(lr_1005_mae))
print('1005 Linear Regression MSE - ' + str(lr_1005_mse))
print()

#######################      Output       #######################

# Sklearn output
joblib.dump(RF_R_1001,'./Output/1001_RandomForest.pkl')
joblib.dump(lr_1001,'./Output/1001_LinearRegression.pkl')
joblib.dump(lr_p_1001,'./Output/1001_Polynomial Regression.pkl')

joblib.dump(RF_R_1002,'./Output/1002_RandomForest.pkl')
joblib.dump(lr_1002,'./Output/1002_LinearRegression.pkl')
joblib.dump(lr_p_1002,'./Output/1002_Polynomial Regression.pkl')

joblib.dump(RF_R_1003,'./Output/1003_RandomForest.pkl')
joblib.dump(lr_1003,'./Output/1003_LinearRegression.pkl')
joblib.dump(lr_p_1003,'./Output/1003_Polynomial Regression.pkl')

joblib.dump(RF_R_1005,'./Output/1005_RandomForest.pkl')
joblib.dump(lr_1005,'./Output/1005_LinearRegression.pkl')
joblib.dump(lr_p_1005,'./Output/1005_Polynomial Regression.pkl')


# Tensorflow
model_1001.save('1001_Tessorflow.h5')
model_1002.save('1002_Tessorflow.h5')
model_1003.save('1003_Tessorflow.h5')
model_1005.save('1005_Tessorflow.h5')