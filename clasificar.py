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
warnings.filterwarnings('ignore') 
participantes = []
#participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

participantes = ['felipe-silva']

clasificaciones = ['knn_1', 'knn_3', 'knn_5', 'knn_8', 'knn_10', 'svmlin_0.1', 'svmlin_1', 'svmlin_10', 'svmlin_100', 'svmPoli2_0.1', 'svmPoli3_0.1', 'svmPoli4_0.1', 'svmPoli5_0.1', 'svmPoli2_1', 'svmPoli3_1', 'svmPoli4_1', 'svmPoli5_1', 'svmPoli2_10', 'svmPoli3_10', 'svmPoli4_10', 'svmPoli5_10', 'svmPoli2_100', 'svmPoli3_100', 'svmPoli4_100', 'svmPoli5_100', 'svmSigm_0.1', 'svmSigm_1', 'svmSigm_10', 'svmSigm_100', 'svmRbf_0.1', 'svmRbf_1', 'svmRbf_10', 'svmRbf_100', 'ann_:2', 'ann_:3', 'ann_' ]

c_acc = [s + '_acc' for s in clasificaciones]
c_acc_std = [s + '_accStd' for s in clasificaciones]
c_prec = [s + '_prec' for s in clasificaciones]
c_prec_std = [s + '_precStd' for s in clasificaciones]
c_rec = [s + '_rec' for s in clasificaciones]
c_rec_std = [s + '_recStd' for s in clasificaciones]
c_f1 = [s + '_f1' for s in clasificaciones]
c_f1_std = [s + '_f1Std' for s in clasificaciones]

clmn = []
for i in range(len(clasificaciones)):
    clmn.append(c_acc[i])
    clmn.append(c_acc_std[i])
    clmn.append(c_prec[i])
    clmn.append(c_prec_std[i])
    clmn.append(c_rec[i])
    clmn.append(c_rec_std[i])
    clmn.append(c_f1[i])
    clmn.append(c_f1_std[i])
    
matrix = np.empty(shape = (len(participantes), len(clmn)))    
i=0    
for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    
    resultados = []
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
  
    caracteristicas = np.array(caracteristicas)
    etiquetas = etiquetas.values.ravel()

     
    #KNN
    print('KNN')
    vecinos = [1, 3, 5, 8, 10]

    for k in vecinos:
        print('vecinos = ' + str(k))
        knn = KNeighborsClassifier(n_neighbors = k)
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, knn, len(repeticion), num_trials=30)
        print('  prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('  prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
      
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
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM polinomial')
    for c in C:
        print('  C = ' + str(c))
        for grado in degree:
            print('   Grado = ' + str(grado))
            svm_poli = SVC(C = c, kernel = 'poly', degree = grado)
            accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_poli, len(repeticion), num_trials=30)
            print('    prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
            print('    prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
            clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM sigmoideo')
    for c in C:
        print('  C = ' + str(c))
        svm_sigm = SVC(C = c, kernel = 'sigmoid')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_sigm, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM rbf')
    for c in C:
        print('  C = ' + str(c))
        svm_rbf = SVC(C = c, kernel = 'rbf')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_rbf, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    #Redes Neuronales
    print('ANN')
    neuronas_ocultas = int(np.sqrt(caracteristicas.shape[0]*len(repeticion)))
    neuronas_o = [neuronas_ocultas - int(neuronas_ocultas/2), neuronas_ocultas - int(neuronas_ocultas/3), neuronas_ocultas]
      
    for neu in neuronas_o:
        print('neuronas ocultas: ' + str(neu))
        ann = MLPClassifier(hidden_layer_sizes = (neu,))
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, ann, len(repeticion), num_trials=30)
        print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    matrix[i, :] = resultados
    i +=1
    
df_resultados = pd.DataFrame(matrix, columns = clmn)

#m = df_resultados.filter(like='_acc') seleccionar maximo y bla