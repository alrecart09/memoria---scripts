#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:31:56 2018

@author: antonialarranaga
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 07:45:45 2017

@author: antonialarranaga
"""
import os
import pandas as pd
import numpy as np
import funciones as fn
import pickle


'''
Se busca el baseline para todas las señales, todos los sujetos: 500 ms desde q dp de relajacion, abren los ojos
'''
def cortar_df(df, df_tpo, timestamp_i, timestamp_f): 
    #timestamps_if es timestamps del eyeTracker con que se desea sincronizar df
    tiempo = np.array(df_tpo)
    indi, indf = fn.sync_indices(tiempo, timestamp_i, timestamp_f)
    df_cortada = fn.cortar_filas_df(indi, indf, df)
    return df_cortada

path = os.path.dirname(os.path.realpath(__file__))

participantes = fn.listaParticipantes()[0]
#participantes = ['diego-gonzalez']
for sujeto in participantes:
    print(sujeto)
    path_ = path + '/sujetos/' + sujeto + '/'
    
    path__ = path_ + 'preproc/'
    path_ventana = path_ + '/ventanas/'
       
    with open(path_ventana + 'relajacion.pkl', 'rb') as f:
        lista_relajacion = pickle.load(f)
       
    timestamp_inicial = lista_relajacion[0][3] + 1000 #ms
    timestamp_final = timestamp_inicial + 500 #ms
       
    señales = ['ppg', 'temp', 'eeg', 'ecg','gsr', 'eyeTracker']
    tiempos = ['ppg_tiempo', 'temp_tiempo', 'eeg_tiempo', 'ecg_tiempo', 'gsr_tiempo', 'eyeTracker_tiempo']
    
    ##cargar señales
    eyeTracker = pd.read_pickle('/Volumes/ADATA CH11/Memoria - scripts2/sujetos/' + sujeto + '/' + sujeto + '_EyeTracker.pkl')
    eyeTracker_tiempo = pd.read_pickle(path_ + 'unix_et')
    eyeTracker['Timestamp_UNIX'] = eyeTracker_tiempo
    
    ppg = pd.read_pickle(path__ + 'ppg.pkl')
    #ppgPeaks = pd.read_pickle(path__ + 'ppgPeaks.pkl')
    ppg_tiempo = ppg['tiempo']
    #ppgPeaks_tiempo = ppgPeaks['tiempo_peaks']
    
    temp = pd.read_pickle(path__ + 'tempPrep.pkl')
    temp_tiempo = temp['tiempo']
    
    eeg = pd.read_pickle(path__ + 'eegPrep.pkl')
    eeg_tiempo = eeg['time']
    
    ecg = pd.read_pickle(path__ + 'ecg.pkl')
    ecg_tiempo = ecg['tiempo']
    #ecgPeaks = pd.read_pickle(path__ + 'ecgPeaks.pkl' )
    #ecgPeaks_tiempo = ecgPeaks['tiempo_peaks']
    
    #gsr 
    gsr = pd.read_pickle(path_ + '/GSR/gsrLedalab.pkl')  
    gsr_tiempo = np.array(gsr['time'])*1000 #ms
    gsr['time'] = gsr_tiempo
    
    date = eyeTracker['RecordingDate'][1] #fecha
    timestamps_eyeTracker = eyeTracker['LocalTimeStamp'] #en hh:mm:ss.ms(3)
    
    baseline = []
    for ind, señal in enumerate(señales):
        relaj = cortar_df(eval(señal), eval(tiempos[ind]), timestamp_inicial, timestamp_final)
        baseline.append(relaj)
        
    path_b = fn.makedir2(path, 'baseline')
    
    with open(path_b + sujeto +'_baseline.pkl', 'wb') as f:
        pickle.dump(baseline, f)
   
   
   

        