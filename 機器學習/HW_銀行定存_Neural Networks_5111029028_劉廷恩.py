# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:46:19 2022

@author: A109021
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:33:20 2022

@author: A109021
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

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


# SVM
from sklearn.svm import SVC

# NN Sk-learn
from sklearn.neural_network import MLPClassifier

# NN Tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Cross-validation
from sklearn.model_selection import cross_val_score

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
#print(df.corr())

final = data.drop(["pdays","previous"],axis=1)

X = final.drop(["y"],axis=1)

Y = final.drop(["age","balance","loan",
                "job","marital","education","default","housing",
                "contact","day","month","duration","campaign",
                "poutcome"],axis=1)
print(Y)
print()

X,Y = make_classification(n_samples=4521,
                          n_features=15,
                          n_informative=2,
                          n_redundant=0,
                          n_clusters_per_class=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25)

# SVM

svc = SVC(C=10,kernel='linear')

svc.fit(X_train, Y_train)
print('SVM score: %.3f' % svc.score(X_test, Y_test))
print('SVM Shape: ' + str(svc.support_vectors_.shape))
print('SVM Cache: ' + str(svc.cache_size))
print()

# NN Sk-learn
mlp = MLPClassifier(
        hidden_layer_sizes=(25,), 
        activation='relu', 
        solver='sgd', 
        learning_rate_init=0.1, 
        max_iter=10000)
mlp.fit(X_train, Y_train)
print('Sk-learn NN Score：' + str(mlp.score(X_test, Y_test)))
print('Sk-learn NN features：' + str(mlp.n_features_in_))
print('Sk-learn NN layer：' + str(mlp.n_layers_))
print('Sk-learn NN epochs：' + str(mlp.n_iter_))
print('Sk-learn NN output：' + str(mlp.n_outputs_))
print()

# NN Tensorflow
model = Sequential()
model.add(Dense(45, activation='relu', input_shape=(15,)))
model.add(Dense(1000, activation='tanh',))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs=30)
#test_los, test_acc = model.evaluate(X_test, Y_test)
print()

# Cross-validation

SVM_scores = cross_val_score(svc, X, Y, cv=5)
print("SVM Cross-validation：" + str(SVM_scores))

# SVM vs NN
# SVM 執行速度較在 NN epoch數不高時執行較慢，但相較 NN 的準確率較為
# 一致，準確率的浮動值較小，但當 NN 的 epoch 提高後，會逐步提高準確
# 率且 NN 可以透過調整 Hiddden Layer 的 Neural 數與 activation 
