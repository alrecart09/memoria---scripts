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

path_nuevo = fn.makedir2(path, 'caracteristicas_ar/' + str(t) + '/')
participantes = ['constantino-hernandez']
for sujeto in participantes:
    
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    

    path_ccs = path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/' 
    nombre_archivo = 'ccs_arousal.pkl' #'ccs_arousal.pkl'
    
    shutil.copy(path_ccs + nombre_archivo, path_nuevo + sujeto +'_' +nombre_archivo)