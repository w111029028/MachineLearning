# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
from sklearn import tree
import numpy as np

data = pd.read_csv("./Source/bank/bank.csv",delimiter=";",header="infer")

print(data.head())

data.age.hist()
plt.title("Test Title")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

final = data.drop(["job","marital","education","default","housing",
                   "contact","day","month","duration","campaign",
                   "pdays","previous","poutcome"],
                  axis=1)

print(final.head())
print()

final.loan.replace(("yes","no"),(1,0), inplace=True)
final.y.replace(("yes","no"),(1,0), inplace=True)
print(final.head())
print()

X = final.drop(["y"],axis=1)
print(X.head())
print()

Y = final.drop(["age","balance","loan"],axis=1)
print(Y.head())
print()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.1)

# 線性資料測試,準確率過低故排除
#lr = LinearRegression()
#lr.fit(X_train, Y_train)
#print(lr.score(X_test, Y_test))
#print(lr.coef_)
#print()

dt1 = tree.DecisionTreeClassifier()
dt1.fit(X_train, Y_train)
print(dt1.score(X_test, Y_test))
print(dt1.n_outputs_)

print( dt1.predict(np.array([52, 2000, 1]).reshape(1, -1)) )
print( dt1.predict(np.array([45, 2000, 1]).reshape(1, -1)) )