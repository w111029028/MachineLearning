# -*- coding: utf-8 -*-
"""
 請以 Linear Regression 以及 Polynomial Regression 來進行預測。
 當然資料前處理要完備，理由以及原則請寫在程式碼的最前面。
 5111029028 劉廷恩
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 正規化工具
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Low Variance Filter
from sklearn.feature_selection import VarianceThreshold

# Feature Importance Filter
from sklearn.feature_selection import SelectKBest, f_regression

# Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# RANSAC
from sklearn.linear_model import RANSACRegressor

# ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel



data = pd.read_csv("./Source/bank/bank.csv",delimiter=";",header="infer")
pd.set_option('max_columns', None)

le = LabelEncoder()

# 資料處理 - 將文字類別轉換為 數字
# 多類別的使用 LabelEncoder 進行，較少類別則手動處理

data['job'] = le.fit_transform(data['job'])
print(data['job'].unique())
data['marital'] = le.fit_transform(data['marital'])
print(data['marital'].unique())
data['education'] = le.fit_transform(data['education'])
print(data['education'].unique())
data.default.replace(("yes","no"),(1,0), inplace=True)
print(data['default'].unique())
data.housing.replace(("yes","no"),(1,0), inplace=True)
print(data['housing'].unique())
data.loan.replace(("yes","no"),(1,0), inplace=True)
print(data['loan'].unique())
data['contact'] = le.fit_transform(data['contact'])
print(data['contact'].unique())
data['month'] = le.fit_transform(data['month'])
print(data['month'].unique())
data['poutcome'] = le.fit_transform(data['poutcome'])
print(data['poutcome'].unique())
data.y.replace(("yes","no"),(1,0), inplace=True)
print(data['y'].unique())


# 資料處理 - 資料正規化 使用 StandardScaler
print(data.head(1))
ss = StandardScaler().fit(data)
ss_scaled_data = ss.fit_transform(data)


# 資料處理 - 篩選 Feature - 排除 High Correlation
# 排除 相關性大於 0.5 的項目
# 13 <--> 14、15 (pdays)
# 14 <--> 13、15 (previous)
# 15 <--> 13、14 (poutcome) -> 15 與其他兩者相關最高故保留 15 其餘刪除

df = pd.DataFrame(ss_scaled_data)
print(df.head(10))
#print(df.corr())

final = data.drop(["pdays","previous"],axis=1)

# Linear Regression

X = final.drop(["y"],axis=1)

Y = final.drop(["age","balance","loan",
                "job","marital","education","default","housing",
                "contact","day","month","duration","campaign",
                "poutcome"],axis=1)
print(Y.head())
print()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3)

lr = LinearRegression(normalize=True)

lr = LinearRegression()
lr.fit(X_train, Y_train)
print(lr.score(X_test, Y_test))
print(lr.coef_)
print()

# RANSAC
rs = RANSACRegressor(lr)
#rs.fit(X, Y)
#rs.score(X, Y)
#rs.estimator_.intercept_
#rs.estimator_.coef_

# 使用 ElasticNet 進行正規化， alphas 越小越好，l1_ratio 越大越好

en = ElasticNet(alpha=0.001, l1_ratio=0.8, normalize=True)
en_scores = cross_val_score(en, X, Y, cv=7, scoring='r2')
encv = ElasticNetCV(alphas=(0.1, 0.01, 0.005,
                            0.001, 0.0005),
                    l1_ratio=(0.1, 0.25, 0.5,
                              0.75, 0.8, 0.9,
                              0.99), normalize=True)
encv.fit(X, Y)
print(encv.alpha_)
print(encv.l1_ratio_)
print(encv.score(X, Y))
print()

# Polynomial Regression

pf = PolynomialFeatures(degree=2)
Xp = pf.fit_transform(X)

#print(Xp.shape)
#print(Y.shape)

lr.fit(Xp, Y)
print(lr.score(Xp, Y))
print()
