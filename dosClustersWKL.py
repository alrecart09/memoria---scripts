#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:37:45 2017

@author: antonialarranaga
"""
import os
import pandas as pd
import funciones as fn
import collections
import  numpy as np
#quedarse con 2 clusters para WKL - mas grandes/diferentes

path = os.path.dirname(os.path.realpath(__file__))

t = 2
#participantes = fn.listaParticipantes()[0]
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

clusters = [[2,5],[2,4],[0,1],[1,0],[2,5],[2,1],[0,5],[1,2],[1,0],[1,0],[2,1],[2,1], [2,5], [1,2], [2,0], [1,0]]

nDatos = []
num = 0         
for sujeto in participantes:
    path_nuevo = fn.makedir2(path , 'dosClustersWKL/')
    
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    
    path_ccs = path +'/caracteristicas_wkl/'+ str(t) + '/' 
    caracteristicas = pd.read_pickle(path_ccs + sujeto + '_ccs.pkl')
    
    #path_etiquetas = path +'/clusters/'+ str(t) + '/'
    path_etiquetas =  path +'/clusters/wkl/'+ str(t) + '/'
    etiquetas = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-wklPupila.pkl')
    
    ccs_eeg =  pd.read_pickle(path + '/caracteristicas_wkl/wkl_nuevasEEG_Zarjam/' + sujeto + '_ccsWkl.pkl')
    ccs_wkl = caracteristicas.drop(['promPupila', 'varPupila'], axis = 1)
    #ccs_wkl['etiquetas'] = etiquetas_wkl
    ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)
    
    etiquetas.reset_index(inplace = True, drop = True)
    ccs_wkl.reset_index(inplace = True, drop = True)
    
    df = pd.concat([etiquetas, ccs_wkl], axis = 1)
    df.columns = ['etiquetas'] + list(ccs_wkl.columns)
    
    clase1 = clusters[num][0]
    clase2 = clusters[num][1]
    
    final =  collections.Counter(df['etiquetas'].values.ravel())
    nClases = len(final)
    
    if nClases>2:
        print('solo dos mayores')
        print(str(clase1), str(clase2))
        df = df[(df['etiquetas'] == clase1) | (df['etiquetas'] == clase2)]
    
    nDatos.append(len(df))
    num+=1
    
    df.reset_index(inplace = True, drop = True)
    df.to_pickle(path_nuevo + sujeto + '_ccsEt.pkl')
    
nDatos_ = pd.DataFrame(nDatos)
print('nDatos ' + str(nDatos_.mean()))