#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:32:32 2018

@author: antonialarranaga
"""

#cantidad de datos promedio
import os
import funciones as fn
import pandas as pd
import numpy as np
#cantidad de clusters en cada etapa de la escritura del ensayo
 
path = os.path.dirname(os.path.realpath(__file__))

path_ = '/Volumes/ADATA CH11/señales_baseline/ccs_todosWKL.pkl'
t = 2 #valencia y arousal
#t= 2 #wkl
#wkl = True

participantes = fn.listaParticipantes()[0]

participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

path_clusters = path + '/clusters/' + str(t) + '/'
i = 0
clmn = ['cant_obs']
matriz = np.empty(shape = (len(participantes), len(clmn)))

i=0
for sujeto in participantes:
    print(sujeto)
    ccs = pd.read_pickle(path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/ccs.pkl')
    cant_obs = len(ccs)
    matriz[i, :] = cant_obs
    i+=1

prom = matriz.mean()
std = matriz.std()