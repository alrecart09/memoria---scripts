#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 18:26:51 2017

@author: antonialarranaga
"""

import os 
import preprocesar
import pandas as pd
import numpy as np
import funciones as fn


#hacer script para pre-procesar (limpiar y filtrar)
show = True
path = os.path.dirname(os.path.realpath(__file__))
path_db = path +'/sujetos/'

participantes = fn.listaParticipantes()[0]
#participantes2 = participantes[49:]

for sujeto in participantes:
#sujeto = 'Pia-Cortes'

    nombre = '/sujetos/' + sujeto + '/'
    path_ = path + nombre
    
    print(sujeto)
    path_prep = fn.makedir(sujeto, path, 'preproc')
    
    #ppg 
    ppg_limpia, timestamps, ppg_amplitud_peaks, ppg_tiempo_peaks = preprocesar.ppg_(path_, sujeto, show = False)
    df_ppg_preproc = pd.DataFrame({'data': ppg_limpia, 'tiempo': timestamps})
    df_ppg_peaks = pd.DataFrame({'amp_peaks': ppg_amplitud_peaks, 'tiempo_peaks':ppg_tiempo_peaks})
    #df_ppg_preproc.to_pickle(path_prep + 'ppgPrep.pkl')
   # df_ppg_peaks.to_pickle(path_prep + 'ppgPeaks.pkl')
    
    #gsr en matlab - LedaLab
    
    #temperatura
    temp_limpia, timestamps = preprocesar.temp_(path_, sujeto, show = False)
    df_temp_preproc = pd.DataFrame({'data': temp_limpia, 'tiempo': timestamps})
    #df_temp_preproc.to_pickle(path_prep + 'tempPrep.pkl')
    
    #eyeTracker dp  - por ventana
    
    #EEG
    hampeleeg, timestamps, ch_nombre = preprocesar.eeg_(path_, sujeto, show = False)
    ch_nombre.append('time')
    signal = np.c_[hampeleeg, timestamps]
    df_eeg_preproc = pd.DataFrame(signal, columns = ch_nombre)
    #df_eeg_preproc.to_pickle(path_prep + 'eegPrep.pkl')
    
    
    #ECG
    filtrada, timestamps, tiempo_peaks, amplitud_peaks = preprocesar.ecg_(path_, sujeto, show = False)
    df_eeg_preproc = pd.DataFrame({'data': filtrada, 'tiempo':timestamps})
    #df_eeg_preproc.to_pickle(path_prep + 'ecgPrep.pkl')
    df_peaksHR = pd.DataFrame({'peaks': amplitud_peaks, 'tiempo_peaks': tiempo_peaks})
    #df_peaksHR.to_pickle(path_prep + 'ecgPeaks.pkl')

