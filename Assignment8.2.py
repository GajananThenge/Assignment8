# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 22:24:26 2018

@author: hp-pc
"""

try:
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from patsy import dmatrices
    from sklearn.linear_model import LogisticRegression
    from sklearn.cross_validation import train_test_split
    from sklearn import metrics
    dta = sm.datasets.fair.load_pandas().data
    from sklearn.metrics import confusion_matrix,accuracy_score
    # add "affair" column: 1 represents having affairs, 0 represents not
    dta['affair'] = (dta.affairs > 0).astype(int)
    y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + \
    religious + educ + C(occupation) + C(occupation_husb)',
    dta, return_type="dataframe")
    
    X = X.rename(columns = {'C(occupation)[T.2.0]':'occ_2',
    'C(occupation)[T.3.0]':'occ_3',
    'C(occupation)[T.4.0]':'occ_4',
    'C(occupation)[T.5.0]':'occ_5',
    'C(occupation)[T.6.0]':'occ_6',
    'C(occupation_husb)[T.2.0]':'occ_husb_2',
    'C(occupation_husb)[T.3.0]':'occ_husb_3',
    'C(occupation_husb)[T.4.0]':'occ_husb_4',
    'C(occupation_husb)[T.5.0]':'occ_husb_5',
    'C(occupation_husb)[T.6.0]':'occ_husb_6'})
    y = np.ravel(y)
    
    #Spliiting tyrain and Test data 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25,random_state=0)
    
    #Create model
    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    
    #Get the Accuracy and confusion matrix
    cm = confusion_matrix(y_test,y_pred)
    acc = accuracy_score(y_test,y_pred)
    
    print("Accutracy in  % : {} ".format(acc.round(3)*100))
except Exception as e:
    print(e)
