#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 22:38:10 2017

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
#combinacion de sensores

t=5 #valencia y arousal
t=2 #wkl

path = os.path.dirname(os.path.realpath(__file__))

t = 2

participantes = fn.listaParticipantes()[0]

num_repeticiones = 5
warnings.filterwarnings('ignore') 
#participantes = []

participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

#participantes = ['ivania-valenzuela']

valencia = False
arousal = True
wkl = False

gsr = True
ppg = True
ecg = True
eyeT = True

if valencia:
    ppg = False
    ecg = False
    sensores = ['eeg', 'temp', 'gsr', 'eyeT']
elif arousal:
    gsr = False
    sensores = ['eeg', 'temp', 'ppg', 'ecg', 'eyeT']
else: #wkl
    eyeT = False 
    participantes = participantes_wkl
    sensores = ['eeg', 'temp', 'gsr', 'ppg', 'ecg']
    
c_acc = [s + '_acc' for s in sensores]
c_acc_std = [s + '_accStd' for s in sensores]
c_prec = [s + '_prec' for s in sensores]
c_prec_std = [s + '_precStd' for s in sensores]
c_rec = [s + '_rec' for s in sensores]
c_rec_std = [s + '_recStd' for s in sensores]
c_f1 = [s + '_f1' for s in sensores]
c_f1_std = [s + '_f1Std' for s in sensores]

clmn = []
for i in range(len(sensores)):
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

participantes = ['boris-suazo']
for sujeto in participantes:
    pupila = False
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    
    if any(sujeto in s for s in participantes_wkl):
        pupila = True

    path_ccsA = path +'/caracteristicas_ar/'+ str(t) + '/' 
    path_ccsV = path +'/caracteristicas_val/'+ str(t) + '/' 
    path_ccs = path+ '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    
    path_etiquetas = path +'/clusters/'+ str(t) + '/'
    
    if wkl:
        eeg =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
        #ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
        #ccs_wkl =  pd.concat([ccs_wkl, eeg], axis=1)
        path_etiqueta = path +'/sujetos/'+ sujeto + '/etiquetas-wklPupila_' + str(t) + '.pkl'
        etiquetas = pd.read_pickle(path_etiqueta)
    elif arousal:  
        eeg  = pd.read_pickle(path_ccsA + sujeto + '_ccs_arousal.pkl')
        #ccs_arousal = ccs.drop(['gsrAcum', 'promGSR', 'powerGSR', 'maxFasica', 'numPeaksFasica', 'promFasica'], axis = 1)#eliminar GSR
        #ccs_arousal = pd.concat([ccs_arousal, eeg], axis = 1)
        etiquetas = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-arousalGSR.pkl') 
    else: #valencia
        eeg = pd.read_pickle(path_ccsV + sujeto + '_ccs_valencia.pkl')
        #ccs_valencia = ccs.drop(['promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin'], axis = 1) #eliminar HR, PPG y ECG 
        #ccs_valencia = pd.concat([ccs_valencia, eeg], axis = 1)
        etiquetas = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-valenciaHR.pkl') 
    
    ccs_eeg = eeg
    ccs_gsr = ccs[['gsrAcum', 'promGSR', 'powerGSR', 'maxFasica', 'numPeaksFasica', 'promFasica']]
    ccs_ppg = ccs[['ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin']]
    ccs_temp = ccs[['pendienteTemp', 'promTemp', 'medianaTemp']]
    ccs_ecg = ccs[['promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD']]
    
    if pupila:
        ccs_eyeT = ccs[['promPupila', 'varPupila', 'numSacadas', 'numFijaciones']]
    else:
        ccs_eyeT = ccs[['numSacadas', 'numFijaciones']]
        
    resultados = []
            
    ccs_eeg.reset_index(drop = True, inplace = True) #tengan mismos indices - partan de 0 hasta len
    ccs_gsr.reset_index(drop = True, inplace = True)
    ccs_ppg.reset_index(drop = True, inplace = True)
    ccs_temp.reset_index(drop = True, inplace = True)
    ccs_ecg.reset_index(drop = True, inplace = True)
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
            ccs_eeg = ccs_eeg.drop(list(indices))
            ccs_gsr.drop(list(indices))
            ccs_ppg.drop(list(indices))
            ccs_temp.drop(list(indices))
            ccs_ecg.drop(list(indices))
            ccs_eyeT.drop(list(indices))
            etiquetas = etiquetas.drop(list(indices))
            
            ccs_eeg.reset_index(drop = True, inplace = True)
            ccs_gsr.reset_index(drop = True, inplace = True)
            ccs_ppg.reset_index(drop = True, inplace = True)
            ccs_temp.reset_index(drop = True, inplace = True)
            ccs_ecg.reset_index(drop = True, inplace = True)
            ccs_eyeT.reset_index(drop = True, inplace = True)
            etiquetas.reset_index(drop = True, inplace = True)
            #print('borre clase ' + str(clase))
    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    
    if len(repeticion) == 1: #revisar que queden más de una clase
        print('no se puede hacer clasificacion, quedó una clase :(')
        continue
    
    print('numero final de clases: ' + str(len(repeticion)))      
  
    ccs_eeg = np.array(ccs_eeg)
    ccs_gsr = np.array(ccs_gsr)
    ccs_ppg = np.array(ccs_ppg)
    ccs_temp = np.array(ccs_temp)
    ccs_ecg = np.array(ccs_ecg)
    etiquetas = etiquetas.values.ravel()
    
    
    if arousal:
        clasificador = SVC(C=10, kernel = 'rbf')#arousal
    elif valencia:
        clasificador = KNeighborsClassifier(n_neighbors = 3) #valencia
    else: #wkl
        clasificador = SVC(C=10, kernel = 'rbf')#wkl
    
    #clasificacion eeg
    print(' eeg')
    accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_eeg, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
    clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    
    #clasificacion temp
    print(' temp')
    accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_temp, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
    clsf.guardar_resultados(accuracy, precision, recall, f1, resultados) 
    
    if gsr:
        print(' gsr')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_gsr, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    if ppg:
        print(' ppg')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_ppg, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
    if ecg:
        print(' ecg')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_ecg, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados) 
    if eyeT:
        print(' eyeT')
        accuracy, precision, recall, f1 = clsf.funcion_clasificar(ccs_eyeT, etiquetas, clasificador, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)        
 
    matrix[i, :] = resultados
    
    i +=1
    
df_resultados = pd.DataFrame(matrix, columns = clmn)

path_resultados = path + '/resultados/'
'''
if valencia:
    df_resultados.to_pickle(path_resultados + 'valencia_clasificadores_porSensor.pkl')
elif arousal:
    df_resultados.to_pickle(path_resultados + 'arousal_clasificadores_porSensor.pkl')
else: #wkl
    df_resultados.to_pickle(path_resultados + 'wkl_clasificadores_porSensor.pkl')
'''    
#m = df_resultados.filter(like='_acc') seleccionar maximo y bla