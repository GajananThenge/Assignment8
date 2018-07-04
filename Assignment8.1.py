# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:35:33 2018

@author: hp-pc
"""

'''Build the linear regression model using scikit learn in boston data to predict
'Price'
based on other dependent variable.
Here is the code to load the data
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
bos = pd.DataFrame(boston.data)
NOTE:​ ​The​ ​solution​ ​shared​ ​through​ ​Github​ ​should​ ​contain​ ​the​ ​source​ ​code​ ​used​ ​
and​ ​the​ ​screenshot​ ​of​ ​the​ ​output.'''

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston

boston = load_boston()
bos = pd.DataFrame(boston.data, columns=boston["feature_names"])

bos['PRICE'] = boston.target

X = bos.iloc[:, :-1].values
y = bos.iloc[:, -1].values

# SImple linear regression handles the scaling internallly
# So skipping
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Create the LInear REgression Model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

plt.plot(y_test, label='Actual Values')
plt.plot(y_pred, label='Predicted Values')
plt.legend()
