#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:59:13 2017

@author: antonialarranaga
"""

'''
Separar ventanas no uniformes de cada sujeto en ventanas uniformes de t segundos (t entre 1 y 5) - para
fracciones de eso, minimo 500 ms. SOLO DEL ENSAYO - no relajación, ni cuestionario, ni instrucciones
'''

import os
import funciones as fn
import pickle


path = os.path.dirname(os.path.realpath(__file__))

participantes = fn.listaParticipantes()[0]
t = 2 #segundos - duracion ventana
minimo = 0.5 #segundos - minimo para ser ventana
cant_señales = 6 #ppg, temp, eeg, ecg, gsr, eyet
fs = [120, 50, 128, 100, 10, 120] #ppg, temp, eeg, ecg, gsr, eyeT

participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'catalina-astorga', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


for sujeto in participantes:
#sujeto = 'pia-cortes' 
    print(sujeto)
    num = 0 #nombre de cada archivo
    
    path_ventana = path +'/señales_baseline/'+ sujeto + '/ventanas/'
    path_ventanasU = fn.makedir2(path +'/señales_baseline/'+ sujeto , 'ventanasU/' + str(t))

    listaVentanas = fn.listaVent(sujeto, '/ventanas/') #igual para con y sin baseline

    for ventana in listaVentanas[:-3]:
    
        with open(path_ventana + ventana, 'rb') as f:
           lista_ventana = pickle.load(f)

        actividad = lista_ventana[0]
        duracion_actividad = (lista_ventana[1][3] - lista_ventana[1][2])/1000 #segundos
        cant_ventanas = int(duracion_actividad/t)
        duracion_residuo = duracion_actividad - cant_ventanas*t
        ventanita_t = []
        residuo= 0 #indica si hubo o no residuo para esa señal
        
        if cant_ventanas > 0:
            for i, señales in enumerate(lista_ventana[2:]): #for i, señales in enumerate(lista_ventana[2:]):
                #señales = lista_ventana[5]    
                muestras = fs[i]*t #cantidad de muestras por ventana
                
                
                for j in range(cant_ventanas): #j va desde el cero a cant_ventanas-1
                    
                    ventanita_señal = fn.cortar_filas_df(j*muestras, (j+1)*muestras, señales)  #señales es un df
                    ventanita_t.append(ventanita_señal) #para cada señal una lista con sus ventanitas
                
                if duracion_residuo > minimo:
                    residuo = 1
                    ventanita_señal = fn.cortar_filas_df((j+1)*muestras, señales.size -1, señales)
                    ventanita_t.append(ventanita_señal)
                
            
            #ventanita_t tiene todas las ventantas de t segundos de todas las señales de una ventana de actividad
            #vent = una ventana de t segundos, todas las señales
            if residuo:
                cant_ventanas +=1
            
            ventanas_t = []
              
            for i in range(cant_ventanas):
                vent = []
                vent.append([actividad])
                for j in range(cant_señales):
                    vent.append(ventanita_t[cant_ventanas*j + i])
                
                with open(path_ventanasU + str(num) +'.pkl', 'wb') as f:
                    pickle.dump(vent, f)
                num +=1
            
        
        
       






