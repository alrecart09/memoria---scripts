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
###extraccion caracteristicas basado en etiquetas de clustering
path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

participantes = []
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

participantes = []
participantes = ['rodrigo-chi']

for sujeto in participantes:
    print(sujeto)

    path_etiquetas = path +'/sujetos/'+ sujeto + '/' 
    etiquetas_wkl =  pd.read_pickle(path_etiquetas + 'etiquetas-wklPupila.pkl')
    
    path_ccs = path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) +  '/' 
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    ccs_eeg =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
    ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    #ccs_wkl['etiquetas'] = etiquetas_wkl
    
    ccs_wkl =  pd.concat([ccs_wkl, ccs_eeg], axis=1)

    selector = sc.rfecvRF(ccs_wkl, etiquetas_wkl)
    
    mascara = selector.support_ #dice cuales elige
    
    seleccion = ccs_wkl.iloc[:, mascara]
    
    print(seleccion.columns)
    
    seleccion.to_pickle(path_ccs +  'seleccion_ccs_wkl.pkl')