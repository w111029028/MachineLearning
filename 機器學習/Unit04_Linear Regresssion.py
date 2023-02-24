# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:28:24 2022

@author: A109021
"""

from sklearn.datasets import load_boston
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


boston = load_boston()

print(boston.data.shape)

df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.plot(subplots=True, layout=(5,3))

X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, test_size=0.1)

lr = LinearRegression(normalize=True)

lr.fit(X_train, Y_train)

print(lr.intercept_)
print(lr.coef_)