#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 21:25:01 2017

@author: antonialarranaga
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocesar


path = os.path.dirname(os.path.realpath(__file__))
#graficos cap 3 - preprocesamiento

participantes = ['lerko-araya']

for sujeto in participantes:

#sujeto = 'Pia-Cortes'

    nombre = '/sujetos/' + sujeto + '/'
    path_ = path + nombre
    
    print(sujeto)
    
    #ppg 
    ppg_limpia, timestamps, ppg_amplitud_peaks, ppg_tiempo_peaks = preprocesar.ppg_(path_, sujeto, show = True)

    
    #gsr en matlab - LedaLab
    
    #temperatura
    #temp_limpia, timestamps = preprocesar.temp_(path_, sujeto, show = True)

    
    #eyeTracker dp  - por ventana
    
    #EEG
    #hampeleeg, timestamps, ch_nombre = preprocesar.eeg_(path_, sujeto, show =True)

    
    
    #ECG
  #  filtrada, timestamps, tiempo_peaks, amplitud_peaks = preprocesar.ecg_(path_, sujeto, show = True)

    #gsr
     #gsr 
    '''
    path_archivo = path_ + sujeto + '_syncGSRPPG.pkl'
    sync_gsrPPG = pd.read_pickle(path_archivo)
    
    data = sync_gsrPPG['GSR_PPG_PPG_A13_CAL']
    
    tpo = sync_gsrPPG['GSR_PPG_TimestampSync_Unix_CAL']
    
    gsr = pd.read_pickle(path_ + '/GSR/gsrLedalab.pkl')  
    tpo_ = sync_gsrPPG['GSR_PPG_TimestampSync_Unix_CAL']
    tpo = gsr['time']
    plt.figure()
    plt.suptitle('Procesamiento señal GSR')
    #ax0 = plt.subplot(311)
    #ax0.plot(tpo_ - tpo_[0], sync_gsrPPG['GSR_PPG_GSR_CAL'], label = 'data GSR original' )
    #ax0.grid()
    #ax0.legend()
    #plt.xlabel('Muestras')
    #plt.ylabel('Resistencia [kohm]')
    ax1 = plt.subplot(211)
    ax1.plot(tpo - tpo[0], gsr['conductance'], 'b-', label='data GSR')
    ax1.grid()
    ax1.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Resistencia')
    ax2 = plt.subplot(212, sharex = ax1)
    ax2.plot(tpo - tpo[0], gsr['fasica'], 'k-', label = 'señal fasica')
    ax2.plot(gsr['tiempo_peaks'] - tpo[0], gsr['peaks_fasica'], 'r.', label = 'peaks')
    ax2.grid()
    ax2.legend()
    plt.xlabel('Muestras')
    plt.ylabel('Resistencia')
    plt.show()
    
    '''

