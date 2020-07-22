#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 19:00:50 2020

@author: vishwa
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib qt

data = pd.read_csv('AutoInsurance.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


plt.scatter(X,y,color = 'red')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 0)

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
sRegressor = LinearRegression()
sRegressor.fit(x_train,y_train)



y_pred = sRegressor.predict(x_test)

#Visualising Simple Linear Model
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')
plt.show()

sRegressor.score(x_test,y_test)
