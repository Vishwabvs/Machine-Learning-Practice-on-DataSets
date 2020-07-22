#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:26:01 2020

@author: vishwa
"""


#Polynomial Regression

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



from sklearn.preprocessing import PolynomialFeatures
pRegressor = PolynomialFeatures(degree =5)
x_train_poly = pRegressor.fit_transform(x_train)
x_test_poly = pRegressor.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
lRegressor = LinearRegression()
lRegressor.fit(x_train_poly,y_train)

y_pred = lRegressor.predict(pRegressor.fit_transform(x_test))

#Visualising training set results
X_grid = np.arange(min(x_train),max(x_train),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(x_train,y_train,color = 'red')
plt.plot(X_grid,lRegressor.predict(pRegressor.fit_transform(X_grid)),color = 'blue')
plt.xlabel('no of claims')
plt.ylabel('total playment for all claims')
plt.show()


#Visualising Test set results
X_grid = np.arange(min(x_test),max(x_test),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(x_test,y_test,color = 'red')
plt.plot(X_grid,lRegressor.predict(pRegressor.fit_transform(X_grid)),color = 'blue')
plt.xlabel('no of claims')
plt.ylabel('total playment for all claims')
plt.show()



lRegressor.score(pRegressor.fit_transform(x_test),y_test)



