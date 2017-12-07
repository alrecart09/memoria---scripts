#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:28:21 2017

@author: antonialarranaga
"""

import funciones as fn
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np


#The question could be rephrased: from the above classification report, how do you output one global number for the f1-score? You could:
#Take the average of the f1-score for each class: that's the avg / total result above. It's also called macro averaging.
#Compute the f1-score using the global count of true positives / false negatives, etc. (you sum the number of true positives / false negatives for each class). Aka micro averaging.
#Compute a weighted average of the f1-score. Using 'weighted' in scikit-learn will weigh the f1-score by the support of the class: the more elements a class has, the more important the f1-score for this class in the computation.      
  


def funcion_clasificar(x, y, clasificador, nClases, num_trials=30):
        
    if nClases == 2:
        scoring = ['accuracy', 'recall', 'precision', 'f1']
        
    else:
        scoring = ['accuracy', 'recall_macro', 'precision_macro', 'f1_macro']

    #scores = cross_validate(svm, caracteristicas, etiquetas, scoring = scoring, cv = 5, n_jobs =-1, return_train_score = True)
    accuracy_i= []
    precision_i= []
    recall_i= []
    f1_i = []
    for i in range(num_trials):
        skf = StratifiedKFold(n_splits=3, shuffle = True)
        score =cross_validate(clasificador, x, y, scoring = scoring, cv = skf, return_train_score = False)
    
        accuracy_i.append(score['test_%s'%scoring[0]].mean())
        precision_i.append(score['test_%s'%scoring[1]].mean())
        recall_i.append(score['test_%s'%scoring[2]].mean())
        f1_i.append(score['test_%s'%scoring[3]].mean())
    
    ac = [np.array(accuracy_i).mean(), np.array(accuracy_i).std()]
    pre = [np.array(precision_i).mean(), np.array(precision_i).std()]
    rec =[np.array(recall_i).mean(), np.array(recall_i).std()] 
    f =  [np.array(f1_i).mean(), np.array(f1_i).std()]
    return ac, pre, rec, f