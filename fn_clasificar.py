#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:28:21 2017

@author: antonialarranaga
"""

import funciones as fn
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from scipy import stats
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
#The question could be rephrased: from the above classification report, how do you output one global number for the f1-score? You could:
#Take the average of the f1-score for each class: that's the avg / total result above. It's also called macro averaging.
#Compute the f1-score using the global count of true positives / false negatives, etc. (you sum the number of true positives / false negatives for each class). Aka micro averaging.
#Compute a weighted average of the f1-score. Using 'weighted' in scikit-learn will weigh the f1-score by the support of the class: the more elements a class has, the more important the f1-score for this class in the computation.      
  
class RandomForestClassifierWithCoef(RandomForestClassifier):
    @property
    def feature_importances_(self):
      # print('hola soy RFCC')
      # print(super().feature_importances_)
        return stats.zscore(super().feature_importances_)
    
    #y=etiquetas.values.ravel() #revisar cantidad de etiquetas?
    #rf = RandomForestClassifierWithCoef(n_estimators = 100, oob_score = True, n_jobs=-1)

def funcion_clasificar(x, y, clasificador, nombre, nClases, num_trials=30):

    #scores = cross_validate(svm, caracteristicas, etiquetas, scoring = scoring, cv = 5, n_jobs =-1, return_train_score = True)
    accuracy_i= []
    precision_i= []
    recall_i= []
    f1_i = []
    for i in range(num_trials):
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, shuffle = True, stratify = y)

        #skf = StratifiedKFold(n_splits=2, shuffle = True)
        rf = RandomForestClassifierWithCoef(n_estimators = 50, oob_score = True, n_jobs=-1)
        rfecv = RFECV(estimator=rf, step=1, cv=2, verbose=0, n_jobs=-1)
        pipe_ = Pipeline([('estandarizar', StandardScaler()), ('rfe-rf', rfecv), (nombre, clasificador)])
        #score =cross_val_score(pipe_rf, x, y, scoring = 'f1', cv = skf)#, return_train_score = False)
        pipe_.fit(xTrain, yTrain)
        yPredichas = pipe_.predict(xTest)
        accuracy = accuracy_score(yTest, yPredichas)
        #print(classification_report(yTest, yPredichas))
        precision, recall, fscore, _ = precision_recall_fscore_support(yTest, yPredichas, average = 'macro')
        
        accuracy_i.append(accuracy)
        precision_i.append(precision)
        recall_i.append(recall)
        f1_i.append(fscore)
    
    ac = [np.array(accuracy_i).mean(), np.array(accuracy_i).std()]
    pre = [np.array(precision_i).mean(), np.array(precision_i).std()]
    rec =[np.array(recall_i).mean(), np.array(recall_i).std()] 
    f =  [np.array(f1_i).mean(), np.array(f1_i).std()]
    return ac, pre, rec, f

def guardar_resultados(accuracy, precision, recall, f1, resultados):
    #print('h')
    resultados.append(accuracy[0])
    resultados.append(accuracy[1])
    resultados.append(precision[0])
    resultados.append(precision[1])
    resultados.append(recall[0])
    resultados.append(recall[1])
    resultados.append(f1[0])
    resultados.append(f1[1])