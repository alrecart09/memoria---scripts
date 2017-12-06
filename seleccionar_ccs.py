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
participantes = ['roberto-rojas']


for sujeto in participantes:
    print(sujeto)
    
    path_etiquetas = path +'/sujetos/'+ sujeto + '/' 
    etiquetas_wkl =  pd.read_pickle(path_etiquetas + 'ccs_wkl-testPupila.pkl')
    
    path_ccs = path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) +  '/' 
    ccs = pd.read_pickle(path_ccs + 'ccs.pkl')
    
    ccs_wkl = ccs.drop(['promPupila', 'varPupila'], axis = 1)
    #ccs_wkl['etiquetas'] = etiquetas_wkl
    
    selector = sc.rfecvRF(ccs_wkl, etiquetas_wkl)
    
    mascara = selector.support_ #dice cuales elige
    
    seleccion = ccs_wkl.iloc[:, mascara]
    
    print('caracteristicas seleccionadas: ' + seleccion.columns)