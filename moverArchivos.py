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

#path_nuevo = fn.makedir2(path, 'caracteristicas_ar/' + str(t) + '/')
participantes = ['catalina-astorga']
path_nuevo = path + 'clusters/' + str(t) + '/'
for sujeto in participantes:
    
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    

    #path_ccs = path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/' 
    #nombre_archivo = 'ccs_valencia.pkl' #'ccs_arousal.pkl'
    
    path_etiquetas = path + '/sujetos/' + sujeto + '/etiquetas-wklPupila_' + str(t) + '.pkl'
    
    shutil.copy(path_etiquetas, path_nuevo + sujeto +'_etiquetas-wklPupila')