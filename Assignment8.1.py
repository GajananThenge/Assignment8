# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:35:33 2018

@author: Gajanan Thenge
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

boston = load_boston()
bos = pd.DataFrame(boston.data, columns=boston["feature_names"])

bos['PRICE'] = boston.target

X = bos.iloc[:, :-1].values
y = bos.iloc[:, -1].values

# Simple linear regression handles the scaling internallly
# So skipping
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42)

# Create the Linear Regression Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predict the Values
y_pred = regressor.predict(X_test)

# The coefficients
print('Coefficients: \n', regressor.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

#To visualize the predicted and actual values
plt.plot(y_test, label='Actual Values')
plt.plot(y_pred, label='Predicted Values')
plt.legend()
