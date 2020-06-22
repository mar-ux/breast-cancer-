# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:16:07 2020

@author: smile
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
da=pd.read_csv("data.csv")
x=da.iloc[:,2:32]
y=da.iloc[:,1]
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X=pca.fit(x)
X=pca.transform(x)
print(X.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=10)
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)

