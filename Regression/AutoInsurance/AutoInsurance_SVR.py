#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 19:36:02 2020

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


from sklearn.svm import SVR
regressor = SVR(kernel = 'linear',degree = 3)
regressor.fit(x_train,y_train)


#Visualising Training Set Results
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'green')
plt.show()


#visualising Test Set Results
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_test,regressor.predict(x_test),color = 'green')
plt.show()


regressor.score(x_test,y_test)
