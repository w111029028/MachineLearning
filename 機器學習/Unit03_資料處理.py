# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 16:00:44 2022

@author: A109021
"""
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer 
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer


boston = load_boston()

#print(boston)


#資料分類處理
# 二分類處理
data = ['male', 'female', 'female', 'male']

#print(np.array([1 if d=='male' 
#          else 0 
#         for d in data]))
#print()
#print(np.array([[1,0] 
#          if d=='male' 
#          else [0,1] 
#          for d in data]))
#print()

# 多分類手動處裡
data1 = ['maybe','yes','no','no','yes','yes']

# print(np.array([1 if d=='yes'
#           else 2 if d=='no'
#           else 3
#           for d in data1]))
# print()
# print(np.array([ [1,0,0] if d=='yes'
#           else [0,1,0] if d=='no'
#           else [0,0,1]
#           for d in data1]))
# print()

# 分類自行處理使用 sklearn - LabelEncoder (非Array)
le = LabelEncoder()
print(le.fit_transform(data1))
print()

# 分類自行處理使用 sklearn - LabelBinarizer (Array)
lb = LabelBinarizer()
print(lb.fit_transform(data1))
print()


# 汲取 具有Target、Value資料 使用 sklearn - DictVectorizer
data3 = [{'height':1, 'lenght':2, 'width':3},
         {'height':4, 'lenght':5, 'width':6},
         {'height':7, 'lenght':8, 'width':9},
         {'height':10, 'lenght':11, 'width':12}]

dv = DictVectorizer(sparse=False)

print(dv.fit_transform(data3))
print()

# Big Mart Data 資料處理

train = pd.read_csv("./Source/Big-Mart-Sales-III-master/Data/Train.csv")
print(train.shape) # 資料結構輸出
print(pd.set_option('display.max_columns', 7)) # 輸出調整：max_columns 欄數，max_rows 行數
print(train.head(1)) # 輸出數量
print()

#  -Exercise 欄位 Item_Fat_Content 處理
train['Item_Fat_Content'] = train['Item_Fat_Content'].str.upper() #大小寫轉換 

train.Item_Fat_Content.replace(("LOW FAT","REGULAR"),("LF","REG"),inplace=True) # 近義詞轉換
print(train['Item_Fat_Content'].unique()) # 找出每欄獨立項目
print(train['Item_Fat_Content'].head(20))  

print(le.fit_transform(train['Item_Fat_Content'])) # LabelEncoder 分類轉換
#print(lb.fit_transform(train['Item_Fat_Content'])) # LabelBinarizer 分類轉換
print()

#  -Exercise 欄位 Item_Type 處理
print(train['Item_Type'].unique())
print(le.fit_transform(train['Item_Type'])) # LabelEncoder 分類轉換
#print(lb.fit_transform(train['Item_Type'])) # LabelBinarizer 分類轉換
print()

# Big Mart Data 資料分析
print("資料分析")
print(train.iloc[5:10]) #輸出特定行數
print(train.isnull().sum()) # 輸出特定條件彙總值
print()

# Missing Data 處理 
#train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)  # 取中位數取代空值
#train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace=True) # 取頻率最高值(眾數)取代空值

#print(imp.fit_transform(train['Outlet_Size']))
train['Item_Weight'].fillna(np.nan,inplace=True)
train['Outlet_Size'].fillna(np.nan,inplace=True)

train_np = train.to_numpy()

print(train_np)

# Missing Data 處理 - Imputer
data = np.array([[1, np.nan, 2],  # 使用 Numpy 的空值 np.nan
                 [2, 3, np.nan], 
                 [-1, 4, 2],
                 [-1, 6, 6]
                ])
#imp = SimpleImputer(missing_values=np.nan, strategy='mean') # 使用 Imputer 調整資料，mean：平均，median：中位數，most_frequent：頻率最高值(眾數)
#print(imp.fit_transform(data)

#data.drop(data.index[(data["y"] == 0)],axis=0,inplace=True)
