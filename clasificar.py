#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:23:23 2017

@author: antonialarranaga
"""

import os
import funciones as fn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd
import collections 
import numpy as np
import warnings
import fn_clasificar as clsf

##CAMBIAR nombre ccs y etiquetas, nombre archivo a guardar

#path = '/Volumes/ADATA CH11'

path = os.path.dirname(os.path.realpath(__file__))


t = 2
#participantes = fn.listaParticipantes()[0]

num_repeticiones = 5
warnings.filterwarnings('ignore') 
#participantes = []

participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

#participantes = ['ivania-valenzuela']
#participantes = ['todos']
path_resultados = fn.makedir2(path, 'resultados/' + str(t) )
clasificaciones = ['knn_1', 'knn_3', 'knn_5', 'knn_10', 'svmlin_1', 'svmlin_10', 'svmlin_100', 'svmPoli2_1', 'svmPoli3_1', 'svmPoli5_1', 'svmPoli2_10', 'svmPoli3_10', 'svmPoli5_10', 'svmPoli2_100', 'svmPoli3_100', 'svmPoli5_100', 'svmSigm_1', 'svmSigm_10', 'svmSigm_100', 'svmRbf_1', 'svmRbf_10', 'svmRbf_100', 'ann:2_0.001', 'ann:2_0.01', 'ann:2_0.1', 'ann_0.001', 'ann_0.01', 'ann_0.1']

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
    
matrix = np.empty(shape = (len(participantes), len(clmn)))    #len(clmn)
i=0    

sujeto = ['alejandro-cuevas']
for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    '''
    path_ccs = path+ '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    path_ccs_ = path + '/caracteristicas_wkl/wkl_nuevasEEG_Zarjam/'
    
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    
    ccs_eeg =  pd.read_pickle(path_ccs_ + sujeto + '_ccsWkl.pkl')
    ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)
    path_etiqueta = path +'/sujetos/'+ sujeto + '/etiquetas-wklPupila_' + str(t) + '.pkl' 
    etiquetasWkl = pd.read_pickle(path_etiqueta)
    '''     
    resultados = []
    
    
  
    path_df = path + '/dosClustersWKL/' + sujeto + '_ccsEt.pkl'
    df = pd.read_pickle(path_df)
    
    etiquetas = df['etiquetas']
    caracteristicas = df.drop(['etiquetas'], axis = 1)

    #etiquetas = etiquetasWkl
    #caracteristicas = ccs_wkl
    #path_ccsA = path +'/caracteristicas_ar/'+ str(t) + '/' 
    #path_ccsV = path +'/caracteristicas_val/'+ str(t) + '/' 
    #path_ccs = path+ '/señales_baseline/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
   # path_ccs = path +'/señales_baseline/ccs_todosWKL.pkl'
    
    #ccs = pd.read_pickle(path_ccs)
    #ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    #ccs_eegA = pd.read_pickle(path_ccsA + sujeto + '_ccs_arousal.pkl')
    #ccs_eegV = pd.read_pickle(path_ccsV + sujeto + '_ccs_valencia.pkl')
    
    #ccs_eeg =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
    #ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    #ccs_wkl['etiquetas'] = etiquetas_wkl
    #ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)
       
    #ccs_arousal = ccs.drop(['gsrAcum', 'promGSR', 'powerGSR', 'maxFasica', 'numPeaksFasica', 'promFasica'], axis = 1)#eliminar GSR
    #ccs_valencia = ccs.drop(['promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin'], axis = 1) #eliminar HR, PPG y ECG 
    
    #ccs_arousal = pd.concat([ccs_arousal, ccs_eegA], axis = 1)
    #ccs_valencia = pd.concat([ccs_valencia, ccs_eegV], axis = 1)
    resultados = []

    #path_etiquetas = path + '/clusters_todosWKL/' + str(t) 
    
    #caracteristicas = ccs_wkl #ccs_a, ccs_v
    #etiquetas = pd.read_pickle(path_etiquetas + '/' + sujeto + 'clusters.pkl') #_etiquetas-arousalGSR.pkl
    #etiquetas = etiquetasWkl
    
    caracteristicas.reset_index(drop = True, inplace = True) #tengan mismos indices - partan de 0 hasta len
    etiquetas.reset_index(drop = True, inplace = True)
    
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
    
    
    clase1 = np.max(etiquetas)
    clase2 = np.min(etiquetas)
    
    etiquetas_ = []
    for etiqueta in etiquetas:
        if etiqueta == clase1:
            etiquetas_.append(1)
        else:
            etiquetas_.append(0)
    etiquetas = np.array(etiquetas_)
    #KNN
    print('KNN')
    vecinos = [1, 3, 5, 10]
    nombre = 'knn'
    for k in vecinos:
        print('vecinos = ' + str(k))
        knn = KNeighborsClassifier(n_neighbors = k)
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, knn, nombre, len(repeticion), num_trials=num_repeticiones)
        #print('  prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        #print('  prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
     
    #SVM
    print('SVM')
    C = [ 1, 10, 100]
    degree = [2, 3, 5] #para poli
    print(' SVM lineal')
    for c in C:
        print('  C = ' + str(c))
        svm_lineal = SVC(C = c, kernel = 'linear')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_lineal, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        #print('    prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        #print('    prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM polinomial')
    for c in C:
        print('  C = ' + str(c))
        for grado in degree:
            print('   Grado = ' + str(grado))
            svm_poli = SVC(C = c, kernel = 'poly', degree = grado)
            accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_poli,'clasificador', len(repeticion), num_trials=num_repeticiones)
            #print('    prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
            #print('    prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
            clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM sigmoideo')
    for c in C:
        print('  C = ' + str(c))
        svm_sigm = SVC(C = c, kernel = 'sigmoid')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_sigm,'clasificador', len(repeticion), num_trials=num_repeticiones)
        #print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        #print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    print(' SVM rbf')
    for c in C:
        print('  C = ' + str(c))
        svm_rbf = SVC(C = c, kernel = 'rbf')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_rbf, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        #print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
        #print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    #Redes Neuronales
    print('ANN')
    neuronas_ocultas = int(np.sqrt(caracteristicas.shape[0]*len(repeticion)))
    neuronas_o = [neuronas_ocultas - int(neuronas_ocultas/2), neuronas_ocultas]
    alpha = [0.001, 0.01, 0.1]
    for neu in neuronas_o:
        print(' neuronas ocultas:' + str(neu))
        for a in alpha:
            print(' alfa: ' + str(a))
            ann = MLPClassifier(hidden_layer_sizes=(neu,), max_iter=400, alpha=a,
                    solver='lbfgs')
            
            accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, ann, 'clasificador', len(repeticion), num_trials=num_repeticiones)
            #print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
            #print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
            clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    matrix[i, :] = resultados
    
    i +=1
    
df_resultados = pd.DataFrame(matrix, columns = clmn)

df_resultados.to_pickle(path_resultados + 'wkl_pupilaBienDosClusters.pkl')
#m = df_resultados.filter(like='_acc') seleccionar maximo y bla