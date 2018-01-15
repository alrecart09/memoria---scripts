#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:26:18 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import shutil

#copiar ccs valencia y excitacion a carpeta general para pasar a otro PC

path = os.path.dirname(os.path.realpath(__file__))
t =5
#participantes = fn.listaParticipantes()[0]

path_nuevo_ccs = fn.makedir2(path, 'caracteristicas_wkl/' + str(t) + '/')
path_nuevo_clusters = fn.makedir2(path, 'clusters/wkl/' + str(t))
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


path_nuevo = path + 'clusters/' + str(t) + '/'
for sujeto in participantes:
    
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    

    path_ccs = path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/' 
    
    path_etiquetas = path + '/sujetos/' + sujeto + '/etiquetas-wklPupila_' + str(t) + '.pkl'
    
    path_ccs_eeg = path_ccs + 'ccs_wkl.pkl'

    
    shutil.copy(path_etiquetas, path_nuevo_clusters + sujeto +'_etiquetas-wklPupila')
    shutil.copy(path_ccs + 'ccs.pkl', path_nuevo_ccs + sujeto + '_ccs.pkl')
    shutil.copy(path_ccs_eeg, path_nuevo_ccs + sujeto + '_ccsEeg.pkl')