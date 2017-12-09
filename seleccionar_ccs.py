#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 18:31:49 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import pandas as pd
import seleccion_caracteristicas as sc
import numpy as np
###extraccion caracteristicas basado en etiquetas de clustering

## cambiar dimension matriz, nombres ccs/etiquetas, nombre archivo a guardar 
path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

#participantes = []
#participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

#participantes = ['braian-wilhelm', 'luz-ugarte']
matriz = np.empty(shape = (len(participantes), 24)) #arousal 29, valencia 24
num = 0
for sujeto in participantes:
    print(sujeto)
    
    path_ccsA = path +'/caracteristicas_ar/'+ str(t) + '/' 
    path_ccsV = path +'/caracteristicas_val/'+ str(t) + '/' 
    path_ccs = path+ '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    ccs_eegA = pd.read_pickle(path_ccsA + sujeto + '_ccs_arousal.pkl')
    ccs_eegV = pd.read_pickle(path_ccsV + sujeto + '_ccs_valencia.pkl')
    
    ##AROUSAL Y VALENCIA  
    ccs_arousal = ccs.drop(['numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR'],axis = 1) #eliminar GSR
    ccs_valencia = ccs.drop(['promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin'], axis = 1) #eliminar HR, PPG y ECG 
    
    ccs_arousal = pd.concat([ccs_arousal, ccs_eegA], axis = 1)
    ccs_valencia = pd.concat([ccs_valencia, ccs_eegV], axis = 1)
    resultados = []

    path_etiquetas = path +'/clusters/'+ str(t) + '/'
    

    etiquetas_val = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-valenciaHR.pkl') #_etiquetas-arousalGSR.pkl
    etiquetas_ar = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-arousalGSR.pkl')
    
    ##WKL##
    #ccs_eeg =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
    #ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    #ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)
    
    ccs = ccs_valencia#ccs_valencia
    etiquetas = etiquetas_val #etiquetas_val
    suma = np.zeros(shape = (ccs.shape[1]))
    
    for i in range(5):
        selector = sc.rfecvRF(ccs, etiquetas)
               
        mascara = selector.support_ #dice cuales elige
        
        #seleccion = ccs_wkl.iloc[:, mascar   
        seleccion = mascara*1 #true en 1, false en 0
        #print('seleccion = ' + str(seleccion))
        suma += seleccion
        #print('suma = ' + str(suma))
    matriz[num, :] = suma
    print('suma total =' + str(suma))
    num +=1

seleccion = pd.DataFrame(data = matriz, columns = ccs.columns)

total = seleccion.sum(axis = 0)
total = total.sort_values(ascending = False)

path_resultados = path + '/resultados/' + str(t) + '/' 

#seleccion.to_pickle(path_resultados +  'seleccion_ccs_arousalHistograma.pkl')
    
print('5 caracteristicas más seleccionadas \n' + str(total[0:5]))