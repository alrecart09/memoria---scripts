#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 15:32:56 2017

@author: antonialarranaga
"""

'''
script para que peaks queden en timestamp que le corresponde y llenar resto con ceros
'''
import funciones as fn
import os
import pandas as pd
import scipy
import numpy as np
import matplotlib.pyplot as plt

path = os.path.dirname(os.path.realpath(__file__))
participantes = fn.listaParticipantes()[0]


#peaks gsr
for sujeto in participantes:
    print(sujeto)
    gsr = path + '/sujetos/' + sujeto + '/GSR/' + sujeto + '_GSRLedalab.mat'
    dict_gsr = scipy.io.loadmat(gsr)
    
    analisis = dict_gsr['analysis']
    fasica = analisis['phasicData'][0][0]
    fasica = fasica[0,:]
    peaks_fasica = analisis['impulseAmp'][0][0]
    peaks_fasica = peaks_fasica[0,:]
    tiempo_peaks = analisis['impulsePeakTime'][0][0]
    
    data = dict_gsr['data']
    conductance = data['conductance'][0][0]
    conductance = conductance[0,:]
    time = data['time'][0][0]
    timeoff = data['timeoff'][0][0]
    
    tiempo_peaks = tiempo_peaks + timeoff
    tiempo = time + timeoff
    
    tiempo_peaks = tiempo_peaks[0,:]
    tiempo = tiempo[0,:]

    
   # plt.plot(tiempo, fasica)
   # plt.plot(tiempo_peaks, peaks_fasica, 'r.')
    
    #abrir .mat de Ledalab
    
    peaks_t = []
    peaks = []
    tiempo_cerca = []
    for index, peak in enumerate(peaks_fasica):
        tiempo = np.array(tiempo)
        tiempo_cerca.append(fn.takeClosest(tiempo, tiempo_peaks[index]))#para tiempo peaks, numero más cerca en tiempo gsr
    
    ind = 0    
    for timestamp in tiempo: #revisar todos los tiempos de gsr
        if ind < len(tiempo_cerca) and timestamp == tiempo_cerca[ind]:
            peaks_t.append(tiempo_cerca[ind])
            peaks.append(peaks_fasica[ind])
            ind +=1
        else:
            peaks_t.append(0)
            peaks.append(0)
    
    df_gsr = pd.DataFrame({'conductance': conductance, 'time': tiempo, 'fasica': fasica, 'peaks_fasica': peaks, 'tiempo_peaks': peaks_t})
    pd.to_pickle(df_gsr, path + '/sujetos/' + sujeto + '/GSR/' + 'gsrLedalab.pkl')


'''
#peaks ECG - indices de peaks R abrir preproc y guardar dnd mismo, pero junto

for sujeto in participantes:

    print(sujeto + ' ECG')
    ecgP_ = path + '/sujetos/' + sujeto + '/preproc/ecgPeaks.pkl'
    ecg_ = path +  '/sujetos/' + sujeto + '/preproc/ecgPrep.pkl'
    
    ecgPeaks = pd.read_pickle(ecgP_)
    ecg = pd.read_pickle(ecg_)
    
    tiempo_cerca = []
    for timestamps in ecgPeaks['tiempo_peaks']: #todos los tiempos de ecg Peaks
        tiempo_cerca.append(fn.takeClosest(ecg['tiempo'], timestamps))
    
    ecgPeaks_ = ecgPeaks['peaks']
    peaks_tiempo = []
    peaks_ecg = []
    ind = 0
    for tiempo in ecg['tiempo']:
        if ind < len(tiempo_cerca) and tiempo == tiempo_cerca[ind]:
            peaks_tiempo.append(tiempo_cerca[ind])
            peaks_ecg.append(ecgPeaks_[ind])
            ind +=1
            
        else:
            peaks_tiempo.append(0)
            peaks_ecg.append(0)
            
    df_ecg = pd.DataFrame({'data': ecg['data'], 'tiempo': ecg['tiempo'], 'peaks': peaks_ecg, 'tiempo_peaks': peaks_tiempo})
    pd.to_pickle(df_ecg, path + '/sujetos/' + sujeto + '/preproc/ecg.pkl')

    #peaks PPG 
    print(sujeto + ' PPG')
    ppgP_ = path + '/sujetos/' + sujeto + '/preproc/ppgPeaks.pkl'
    ppg_ = path +  '/sujetos/' + sujeto + '/preproc/ppgPrep.pkl'
    
    ppgPeaks = pd.read_pickle(ppgP_)
    ppgPeaks = ppgPeaks.reset_index(drop=True)
    ppg = pd.read_pickle(ppg_)
    
    tiempo_cerca = []
    for timestamps in ppgPeaks['tiempo_peaks']: #todos los tiempos de ecg Peaks
        tiempo_cerca.append(fn.takeClosest(ppg['tiempo'], timestamps))
    
    ppgPeaks_ = ppgPeaks['amp_peaks']

    ppgPeaks_tiempo = []
    ppgPeaks = []
    ind = 0
    for tiempo in ppg['tiempo']:
        if ind < len(tiempo_cerca) and tiempo == tiempo_cerca[ind]:
            ppgPeaks_tiempo.append(tiempo_cerca[ind])
            ppgPeaks.append(ppgPeaks_[ind])
            ind +=1
            
        else:
            ppgPeaks_tiempo.append(0)
            ppgPeaks.append(0)
            
    df_ppg = pd.DataFrame({'data': ppg['data'], 'tiempo': ppg['tiempo'], 'peaks': ppgPeaks, 'tiempo_peaks': ppgPeaks_tiempo})
    pd.to_pickle(df_ppg, path + '/sujetos/' + sujeto + '/preproc/ppg.pkl')
'''