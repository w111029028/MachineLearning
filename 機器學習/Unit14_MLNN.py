# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:09:03 2022

@author: A109021
"""

from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier

nb_samples = 1000
nb_features = 3
X, Y = make_classification(n_samples=nb_samples, n_features=nb_features, n_informative=3, n_redundant=0, n_classes=2, n_clusters_per_class=3)


mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='tanh', solver='sgd', learning_rate_init=0.1, max_iter=10000)
#mlp.fit(X_train, Y_train)
#mlp.score(X_test, Y_test)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

model = Sequential()
model.add(Dense(50, activation='tanh', input_shape=(3,)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])

model.add(Input(shape=(3,)))
model.add(Dense(50, activation='tanh'))
