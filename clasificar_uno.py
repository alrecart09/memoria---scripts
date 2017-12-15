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

path = os.path.dirname(os.path.realpath(__file__))

t = 2
#participantes = fn.listaParticipantes()[0]

num_repeticiones = 5
warnings.filterwarnings('ignore') 
#participantes = []

participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

participantes = ['roberto-rojas']
path_resultados = fn.makedir2(path, 'resultados/' + str(t) )
clasificaciones = [ 'svmRbf_10']

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
for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    
    '''
    path_ccs = path+ '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    
    ccs_eeg =  pd.read_pickle(path + '/caracteristicas_wkl/wkl_nuevasEEG_Zarjam/' + sujeto+ '_ccsWkl.pkl')
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

    #SVM
    print('SVM')
    C = 10
    
    print(' SVM rbf')

    print('  C = ' + str(C))
    svm_rbf = SVC(C = C, kernel = 'rbf')
    accuracy, precision, recall, f1 = clsf.funcion_clasificar(caracteristicas, etiquetas, svm_rbf, 'clasificador', len(repeticion), num_trials=num_repeticiones)
    #print('   prom acc: ' + str(accuracy[0]) + ' std :' + str(accuracy[1]))
    #print('   prom f1: ' + str(f1[0]) + ' std :' + str(f1[1]))
    clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    
    matrix[i, :] = resultados
    
    i +=1
    
df_resultados = pd.DataFrame(matrix, columns = clmn)

#df_resultados.to_pickle(path_resultados + 'arousal_clasificadores_eegSeleccion.pkl')
#m = df_resultados.filter(like='_acc') seleccionar maximo y bla