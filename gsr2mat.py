#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:59:08 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import pickle
import scipy
'''
script para sacar gsr de ventanas uniformes y guardarlos como .mat en ventana GSR 
dentro de ventanasU/t
'''

path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

sujeto = 'ismael-silva'
for sujeto in participantes:
    print(sujeto)
    num = 0 #num ventana
    listaVentanas = fn.listaVent(sujeto, '/ventanasU/' + str(t) + '/')
    
    path_ventana = path +'/sujetos/'+ sujeto + '/ventanasU/' + str(t) +  '/'
    path_gsr = fn.makedir(sujeto, path, 'ventanasU/5/GSR')    
        
    for ventana in listaVentanas:
    
        with open(path_ventana + ventana, 'rb') as f:
           lista_ventana = pickle.load(f)
           
        gsr = lista_ventana[5]
    
        save_gsr ={'conductance': gsr['GSR_PPG_GSR_CAL'].values, 'time': gsr['GSR_PPG_TimestampSync_Unix_CAL'].values}
        scipy.io.savemat(path_gsr + str(num) + '.mat' ,save_gsr)
        num +=1
    