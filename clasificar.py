#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:23:23 2017

@author: antonialarranaga
"""

import os
import funciones as fn
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import collections 
import numpy as np
import warnings
import fn_clasificar as clsf

path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

num_repeticiones = 15
warnings.filterwarnings('always') 
participantes = []
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

#participantes = ['felipe-silva']
for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
       
    path_ccs = path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) +  '/' 
    path_etiquetas = path +'/sujetos/'+ sujeto + '/'
    caracteristicas = pd.read_pickle(path_ccs + 'seleccion_ccs_wkl.pkl')
    etiquetas = pd.read_pickle(path_etiquetas + 'etiquetas-wklPupila.pkl')

    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    print('numero original de clases: ' + str(len(repeticion)))
    
    for clase, cantidad in repeticion:
       # print(str(clase) + ' ' + str(cantidad))
        if cantidad < 6:
            #print('clase ' + str(clase) + ' con <= de 5 elementos')
            
            indices = np.where(etiquetas == clase)[0]
            #print(indices)
            caracteristicas = caracteristicas.drop(list(indices))
            etiquetas = etiquetas.drop(list(indices))
            
            caracteristicas.reset_index(drop = True, inplace = True)
            etiquetas.reset_index(drop = True, inplace = True)
            #print('borre clase ' + str(clase))
    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    
    if len(repeticion) == 1: #revisar que queden más de una clase
        print('no se puede hacer clasificacion, quedó una clase :(')
        continue
    
    print('numero final de clases: ' + str(len(repeticion)))      
    '''    
    caracteristicas = np.array(caracteristicas)
    etiquetas = etiquetas.values.ravel()

     
    #KNN
    print('KNN')
    vecinos = [1, 3, 5, 8, 10]
    accuracy_knn = []
    precision_knn = []
    recall_knn = []
    f1_knn = []
    for k in vecinos:
        print('vecinos = ' + str(k))
        knn = KNeighborsClassifier(n_neighbors = k)
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, knn, len(repeticion), num_trials=30)
        print('  prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('  prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        accuracy_knn.append(accuracy)
        precision_knn.append(precision)
        recall_knn.append(recall)
        f1_knn.append(f1)
        
    #SVM
    print('SVM')
    C = [0.1, 1 ,10, 100]
    degree = [2, 3, 4, 5] #para poli
    print(' SVM lineal')
    for c in C:
        print('  C = ' + str(c))
        svm_lineal = SVC(C = c, kernel = 'linear')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_lineal, len(repeticion), num_trials=30)
        print('    prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('    prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        
    print(' SVM polinomial')
    for c in C:
        print('  C = ' + str(c))
        for grado in degree:
            print('   Grado = ' + str(grado))
            svm_poli = SVC(C = c, kernel = 'poly', degree = grado)
            accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_poli, len(repeticion), num_trials=30)
            print('    prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
            print('    prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
    
    print(' SVM sigmoideo')
    for c in C:
        print('  C = ' + str(c))
        svm_sigm = SVC(C = c, kernel = 'sigmoid')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_sigm, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
    
    print(' SVM rbf')
    for c in C:
        print('  C = ' + str(c))
        svm_rbf = SVC(C = c, kernel = 'rbf')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_rbf, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        
    #Redes Neuronales
    print('ANN')
    neuronas_ocultas = int(np.sqrt(caracteristicas.shape[0]*len(repeticion)))
    neuronas_o = range(5, neuronas_ocultas+5, 5)
        
    for neu in neuronas_o:
        print('neuronas ocultas: ' + str(neu))
        ann = MLPClassifier(hidden_layer_sizes = (neu,))
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, ann, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
    '''    