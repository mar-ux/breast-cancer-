# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:10:48 2020

@author: smile
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
da=pd.read_csv("data.csv")
da.info()
import seaborn as sns
corrmat = da.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(da[top_corr_features].corr(),annot=True,cmap="RdYlGn")
da.hist()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
x=da.iloc[:,2:32]
y=da.iloc[:,1]
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 
featureScores
print(featureScores.nlargest(10,'Score'))
Xnew=pd.read_csv("data.csv",usecols=('area_worst','area_mean','area_se','perimeter_worst','perimeter_mean','radius_worst','radius_mean','perimeter_se','texture_worst','texture_mean'))
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
Xnew= standardScaler.fit_transform(Xnew)
X=Xnew
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X_train,y_train,cv=10)
    knn_scores.append(score.mean())
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
knn_classifier = KNeighborsClassifier(n_neighbors = 14)
knn_classifier.fit(X_train,y_train)
score=cross_val_score(knn_classifier,X_train,y_train,cv=10)
y_pred=knn_classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test, y_pred)
