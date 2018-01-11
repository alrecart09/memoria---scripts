#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 07:45:45 2017

@author: antonialarranaga
"""
import os
import sys
import pandas as pd
import numpy as np
import funciones as fn
import pickle

'''
Este script va sujeto a sujeto recorriendo las etiquetas manuales y cortando cada señal psicofisiologica
segun las etiquetas existentes del eye Tracker - guarda las ventanas con el nombre de la actividad, el tiempo
del eye Tracker y las señales cortadas de acuerdo a ese tiempo
Son ventanas de tiempos no uniformes
'''

#hacer carpeta para guardar preproc - si ya existe no hace nada


def timestamps(date, timestamp1, timestamp2):
    ti = fn.toUnix(date, timestamp1)#pasarlo a unix para sincronizar
    tf= fn.toUnix(date, timestamp2)
    return ti, tf

def cortar_df(df, df_tpo, timestamp_i, timestamp_f): 
    #timestamps_if es timestamps del eyeTracker con que se desea sincronizar df
    tiempo = np.array(df_tpo)
    indi, indf = fn.sync_indices(tiempo, timestamp_i, timestamp_f)
    df_cortada = fn.cortar_filas_df(indi, indf, df)
    return df_cortada


path = os.path.dirname(os.path.realpath(__file__))

participantes = fn.listaParticipantes()[0]

for sujeto in participantes:

#sujeto = 'Pia-Cortes' 
    print(sujeto)
    path__ = path + '/señales_baseline/' + sujeto + '/'
    path_ventanas = fn.makedir2(path__ , 'ventanas')
    
    señales = ['ppg', 'temp', 'eeg', 'ecg','gsr', 'eyeTracker']
    tiempos = ['ppg_tiempo', 'temp_tiempo', 'eeg_tiempo', 'ecg_tiempo', 'gsr_tiempo', 'eyeTracker_tiempo']
    
    ##cargar señales
    eyeTracker = pd.read_pickle(path__ + 'eyeTracker.pkl')
    
    ppg = pd.read_pickle(path__ + 'ppg.pkl')
    #ppgPeaks = pd.read_pickle(path__ + 'ppgPeaks.pkl')
    ppg_tiempo = ppg['tiempo']
    #ppgPeaks_tiempo = ppgPeaks['tiempo_peaks']
    
    temp = pd.read_pickle(path__ + 'temp.pkl')
    temp_tiempo = temp['tiempo']
    
    eeg = pd.read_pickle(path__ + 'eeg.pkl')
    eeg_tiempo = eeg['time']
    
    ecg = pd.read_pickle(path__ + 'ecg.pkl')
    ecg_tiempo = ecg['tiempo']
    
    #gsr 
    gsr = pd.read_pickle(path__ + '/GSR/gsrLedalab.pkl')  
    gsr_tiempo = np.array(gsr['time'])*1000 #ms
    gsr['time'] = gsr_tiempo
    
    date = eyeTracker['RecordingDate'][1] #fecha
    timestamps_eyeTracker = eyeTracker['LocalTimeStamp'] #en hh:mm:ss.ms(3)
    
    
    eyeTracker_tiempo = pd.read_pickle(path + '/sujetos/' + sujeto  + '/unix_et') # =fn.timeStampEyeTracker(date, timestamps_eyeTracker)
    eyeTracker['Timestamps_UNIX'] = eyeTracker_tiempo
    #eyeTracker_tiempo.to_pickle(path_ + 'unix_et') #en siguientes ventanas solo abrirlo
    indices = eyeTracker['StudioEventIndex'] #indices de todos los eventos
    etiquetas_evento = eyeTracker['StudioEventData'].dropna() #sin nans
    tipo_evento = eyeTracker['StudioEvent'].dropna()
    
    indices_eventos = []
    
    #recuperar indices de etiquetas
    indices_tipo = tipo_evento.index.values
    indices_etiquetas = etiquetas_evento.index.values
    
    
    for i in range(0, tipo_evento.shape[0]):
        if tipo_evento[indices_tipo[i]] == 'Default':
            indices_eventos.append(indices_tipo[i])
        
    etiquetas_evento = etiquetas_evento[indices_eventos] #etiquetas solo de eventos que yo anote
    
    if sujeto == 'boris-suazo':
        etiquetas_evento = etiquetas_evento.drop(38365)
    
    indices_etiquetas = etiquetas_evento.index.values   
        
    #obtener timestamps y etiquetas para cortar
    if etiquetas_evento[indices_etiquetas[0]] != 'relajacion_i':
        print(sujeto + ' error rel')
        sys.exit(0)
    
    
    #etiquetas estandar para todos
    r_i=  timestamps_eyeTracker[indices_etiquetas[0]] #indice donde parte la relajacion
    r_f=  timestamps_eyeTracker[indices_etiquetas[1]]
    relajacion_i, relajacion_f = timestamps(date, r_i, r_f) #timestamps de corte para relajacion
    
    
    
    c_i=  timestamps_eyeTracker[indices_etiquetas[2]]
    c_f=  timestamps_eyeTracker[indices_etiquetas[3]]
    cuest_i, cuest_f = timestamps(date, c_i, c_f) #timestamps de corte para cuestionario
    
    i_i=  timestamps_eyeTracker[indices_etiquetas[4]]
    i_f=  timestamps_eyeTracker[indices_etiquetas[5]]
    instr_i, instr_f = timestamps(date, i_i, i_f) #timestamps de corte para instrucciones_ensayo (2da pag enc)
    
    relajacion = []
    relajacion.append([timestamps_eyeTracker[indices_etiquetas[0]],  timestamps_eyeTracker[indices_etiquetas[1]], relajacion_i, relajacion_f])
    cuestionario = []
    cuestionario.append([timestamps_eyeTracker[indices_etiquetas[2]],  timestamps_eyeTracker[indices_etiquetas[3]], cuest_i, cuest_f])
    instrucciones = []
    instrucciones.append([timestamps_eyeTracker[indices_etiquetas[4]],  timestamps_eyeTracker[indices_etiquetas[5]], instr_i, instr_f ])
            
    for ind, señal in enumerate(señales):
        relaj = cortar_df(eval(señal), eval(tiempos[ind]), relajacion_i, relajacion_f)
        relajacion.append(relaj)
        cuest = cortar_df(eval(señal), eval(tiempos[ind]), cuest_i, cuest_f)
        cuestionario.append(cuest)
        instr = cortar_df(eval(señal), eval(tiempos[ind]), instr_i, instr_f)
        instrucciones.append(instr)
    
    with open(path_ventanas +'relajacion.pkl', 'wb') as f:
        pickle.dump(relajacion, f)
    
    with open(path_ventanas +'cuestionario.pkl', 'wb') as f:
        pickle.dump(cuestionario, f)
    
    with open(path_ventanas +'instrucciones.pkl', 'wb') as f:
        pickle.dump(instrucciones, f)
        
    #etiquetas ensayo
    if etiquetas_evento[indices_etiquetas[6]] != 'instr_ens_i':
        print(sujeto + ' error ensayo')
        sys.exit(0)
    
    ventana = 0
    for j, i in enumerate(indices_etiquetas[6:]): #desde el 5 hasta el n-1
        if etiquetas_evento[i] == 'cortar_i':
            continue
            
        #print(etiquetas_evento[i])
        if i == indices_etiquetas[indices_etiquetas.size -1]:
            break
        lista = []
        lista.append(etiquetas_evento[i]) #nombre de la etiqueta
        
        tpo_i =  timestamps_eyeTracker[i]
        tpo_f =  timestamps_eyeTracker[indices_etiquetas[j+7]] #j va desde el 0 a n-1, desfase de 6 + 1 pq es el proximo
        t1, t2 = timestamps(date, tpo_i, tpo_f)
        lista.append([timestamps_eyeTracker[i],timestamps_eyeTracker[indices_etiquetas[j+7]], t1,t2])
        
        for i, señal in enumerate(señales):
            df = cortar_df(eval(señal), eval(tiempos[i]), t1, t2)
            lista.append(df)

        with open(path_ventanas + str(ventana) +'.pkl', 'wb') as f:
            pickle.dump(lista, f)
        ventana = ventana + 1
        
    
#abrir lista 
#with open('parrot.pkl', 'rb') as f:
#   mynewlist = pickle.load(f)