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
import caracteristicas as cc
###extraccion caracteristicas basado en etiquetas de clustering

## cambiar dimension matriz, nombres ccs/etiquetas, nombre archivo a guardar 
path = os.path.dirname(os.path.realpath(__file__))
t = 2

'''
participantes = ['luz-ugarte',
 'manuela-diaz',
 'matias-gomez',
 'matias-mattamala',
 'mauricio-avdalov',
 'melissa-chaperon',
 'michelle-fredes',
 'miguel-sanchez',
 'nicolas-burgos',
 'nicolas-mellado',
 'pablo-gonzalez',
 'patricio-mallea',
 'pia-cortes',
 'ricardo-ramos',
 'roberto-rojas',
 'rodrigo-chi',
 'rodrigo-perez',
 'tom-cataldo',
 'tomas-lagos']
'''
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

#participantes = ['catalina-astorga']
matriz = np.empty(shape = (len(participantes), 43)) #arousal 29, valencia 24
#matriz2 = np.empty(shape = (len(participantes), 26))#arousal 31, valencia 26
num1 = 0
num2=0
conPupila= 26
sinPupila=24
for sujeto in participantes:
    print(sujeto)
    
    path_ccsA = path +'/caracteristicas_ar/'+ str(t) + '/' 
    path_ccsV = path +'/caracteristicas_val/'+ str(t) + '/' 
    path_ccs = path+ '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    ccs_eegA = pd.read_pickle(path_ccsA + sujeto + '_ccs_arousal.pkl')
    ccs_eegV = pd.read_pickle(path_ccsV + sujeto + '_ccs_valencia.pkl')
    
    etiquetas_wkl = pd.read_pickle(path + '/sujetos/' + sujeto + '/etiquetas-wklPupila_' + str(t) + '.pkl')
    '''
    ##AROUSAL Y VALENCIA  
    ccs_arousal = ccs.drop(['numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR'],axis = 1) #eliminar GSR
    ccs_valencia = ccs.drop(['promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin'], axis = 1) #eliminar HR, PPG y ECG 
    
    ccs_arousal = pd.concat([ccs_arousal, ccs_eegA], axis = 1)
    ccs_valencia = pd.concat([ccs_valencia, ccs_eegV], axis = 1)
    resultados = []
    
    #df.isnull().values.any()
    path_etiquetas = path +'/clusters/'+ str(t) + '/'
    

    etiquetas_val = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-valenciaHR.pkl') #_etiquetas-arousalGSR.pkl
    etiquetas_ar = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-arousalGSR.pkl')
    #print(ccs_valencia.shape[1])
    '''
    
    ##WKL##
    ccs_eeg =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
    ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)
    
    ccs = ccs_wkl#ccs_valencia
    etiquetas = etiquetas_wkl #etiquetas_val
    
    ccs = cc.escalar_df(ccs) #sean comparables
    suma = np.zeros(shape = (ccs.shape[1]))
   
    for i in range(5):
        selector = sc.rfecvRF(ccs, etiquetas)
               
        mascara = selector.support_ #dice cuales elige
        
        #seleccion = ccs_wkl.iloc[:, mascar   
        seleccion = mascara*1 #true en 1, false en 0
        #print('seleccion = ' + str(seleccion))
        suma += seleccion
        #print('suma = ' + str(suma))
    matriz[num1, :] = suma
    num1 +=1
    
    '''
    if ccs.shape[1] == conPupila:        
        matriz2[num1, :] = suma
        cols2 = ccs_valencia.columns
        num1 +=1
    else:
        matriz[num2,:] = suma
        cols = ccs_valencia.columns
        num2 +=1
    '''
    print('suma total =' + str(suma))

seleccion = pd.DataFrame(data = matriz, columns = ccs_wkl.columns) 
total = seleccion.sum(axis = 0)
total = total.sort_values(ascending = False)

#seleccion_sinPupila = pd.DataFrame(data = matriz, columns = cols)
#seleccion_conPupila = pd.DataFrame(data = matriz2, columns = cols2)

#total_sinPupila = seleccion_sinPupila.sum(axis = 0)
#total_sinPupila = total_sinPupila.sort_values(ascending = False)

#total_conPupila = seleccion_conPupila.sum(axis = 0)
#total_conPupila = total_conPupila.sort_values(ascending = False)

path_resultados = path + '/resultados/' + str(t) + '/' 

seleccion.to_pickle(path_resultados + 'seleccion_ccs_wklHistograma.pkl')
#seleccion_sinPupila.to_pickle(path_resultados +  'seleccion_ccs_valenciaHistogramaSinPupila_desdeLerko.pkl')
#seleccion_conPupila.to_pickle(path_resultados +  'seleccion_ccs_valenciaHistogramaConPupila_desdeLerko.pkl')
#print('5 caracteristicas más seleccionadas con pupila \n' + str(total_conPupila[0:5]))
print('5 caracteristicas más seleccionadas \n' + str(total[0:5]))
