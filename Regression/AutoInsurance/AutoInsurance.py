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
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Simple Linear Regression
from sklearn.linear_model import LinearRegression
sRegressor = LinearRegression()
sRegressor.fit(x_train,y_train)


sRegressor.score(x_test,y_test)
y_pred = sRegressor.predict(x_test)

#Visualising Simple Linear Model
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,sRegressor.predict(x_train),color = 'blue')
plt.show()



#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
pRegressor = PolynomialFeatures(degree =2)
X_poly = pRegressor.fit_transform(X)



lRegressor = LinearRegression()
lRegressor.fit(X_poly,y_train)

X_polyTest = pRegressor.fit_transform(x_test)
lRegressor.score(x_train,y_test)
y_pred = lRegressor.predict(x_train)

#Visualising Polynomial Model
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,lRegressor.predict(x_train),color = 'blue')
plt.show()




