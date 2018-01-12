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

#participantes = fn.listaParticipantes()[0]
participantes = ['alejandro-cuevas']

participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'catalina-astorga', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


for sujeto in participantes:
    print(sujeto)
    
    path_ = path + '/sujetos/' + sujeto + '/'
    
    path__ = path_ + 'preproc/'
    path_ventana = path_ + '/ventanas/'
       
    eyeT_baseline = pd.read_pickle(path + '/antonia_baseline/' + sujeto + '.pkl')
    
    indices = eyeT_baseline['StudioEventIndex'] #indices de todos los eventos
    etiquetas_evento = eyeT_baseline['StudioEventData'].dropna() #sin nans
    tipo_evento = eyeT_baseline['StudioEvent'].dropna()
    timestamps_eyeTracker = eyeT_baseline['LocalTimeStamp'] #en hh:mm:ss.ms(3)

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

    #etiquetas estandar para todos
    indice = [i for i, x in enumerate(etiquetas_evento) if x == 'baseline'][0]
    print(indice)
    baseline = timestamps_eyeTracker[indices_etiquetas[indice]] #indice donde parte la relajacion
    print(indices_etiquetas[indice])
    date = eyeT_baseline['RecordingDate'][1]
    baseline_timestamp = fn.toUnix(date, baseline)
     
    timestamp_inicial = baseline_timestamp #ms
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
   
   
   

        