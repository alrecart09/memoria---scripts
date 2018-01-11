#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 15:20:29 2018

@author: antonialarranaga
"""

#señal - baseline, normalizada por z-score

import os
import pandas as pd
import numpy as np
import funciones as fn
import pickle
from scipy import interpolate
import mod_filtros as filtro
import math
import scipy.io

#eyeTracker
def diametroPupila_(eyeTracker, show = False):
    
    fs = 120 #hz
    #recuperar datos importantes del df
    datosPupilaIzq = eyeTracker['PupilLeft']
    datosPupilaDer = eyeTracker['PupilRight']
    validezPupilaIzq = eyeTracker['ValidityLeft']
    validezPupilaDer = eyeTracker['ValidityRight']
    
    #timestamps = fn.timeStampEyeTracker(eyeTracker) #lento
    
    datosSacadas = eyeTracker['GazeEventType']
    
    #recuperar datos pupila mas confiable
    diametroPupila = []
    validez = []
    for index, (izq, der) in enumerate(zip(validezPupilaIzq, validezPupilaDer)):
        if izq == 0 and der == 0:
            prom = float((datosPupilaDer[index] + datosPupilaIzq[index])/2)
            diametroPupila.append(prom)
            validez.append(der)
        elif izq<der: #derecha es mas confiable
            diametroPupila.append(datosPupilaDer[index])
            validez.append(der)
        else: #izquierda es mas confiable
            diametroPupila.append(datosPupilaIzq[index])
            validez.append(izq)
    
    validez = np.array(validez)
    diametroPupila = np.array(diametroPupila)
    
    valid = validez[~np.isnan(validez)]     
    valid = sum(valid)/valid.size #promedio de validez de no NANs
    
    tiempo = np.array(eyeTracker['EyeTrackerTimestamp']) #referencial - cambiar por Timestamps_UNIX
    #tiempo_ = tiempo - tiempo[0] 
    
    #sacadas como NAN para dp interpolar 
    for index, tipo in enumerate(datosSacadas): 
        if tipo == 'Saccade':
            diametroPupila[index] = float('nan')
            
    #interpolar NANs (esto hacerlo para cada ventana, pq 5 primeros minutos es de relajacion cn ojos cerrados)
    indicesNAN = np.isnan(diametroPupila)
    
    
    pupila_sinNAN = diametroPupila[~indicesNAN]
    tiempo_sinNAN = tiempo[~indicesNAN]
    
    
    f_inter = interpolate.interp1d(tiempo_sinNAN, pupila_sinNAN) #tipo de interpolación
    tpo = np.linspace(tiempo_sinNAN[0], tiempo_sinNAN[tiempo_sinNAN.size-1], diametroPupila.size)
    pupila_interpolada = f_inter(tpo)
    
    order = int(0.3 * fs)
    pupila_filtrada, _, _ = filtro.filter_signal2(signal=pupila_interpolada,
                                      ftype='FIR',
                                      band='lowpass',
                                      order=order,
                                      frequency=2,
                                      sampling_rate=fs)
       
    return pupila_filtrada, tpo

def baseline_peaks(df, baseline):
    peak_ppg = []
    for peaks in df:
        if peaks != 0:
            peaks = peaks - baseline
            peak_ppg.append(peaks)
        else:
            peaks = 0
            peak_ppg.append(peaks)
    return peak_ppg

def z_score(df): 
    return (df-df.mean())/df.std(ddof=0)
            
path = os.path.dirname(os.path.realpath(__file__))
participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

participantes = fn.listaParticipantes()[0]

participantes = ['alejandro-cuevas']
for sujeto in participantes:
    print(sujeto)
    
    wkl = False
    if any(sujeto in s for s in participantes_wkl):
        wkl = True
        
    path_ = path + '/sujetos/' + sujeto + '/'
    
    path__ = path_ + 'preproc/'
    path_ventana = path_ + '/ventanas/'

              
    #señales = ['ppg', 'temp', 'eeg', 'ecg','gsr', 'eyeTracker']
    #tiempos = ['ppg_tiempo', 'temp_tiempo', 'eeg_tiempo', 'ecg_tiempo', 'gsr_tiempo', 'eyeTracker_tiempo']
    
    ##cargar señales
    eyeTracker = pd.read_pickle('/Volumes/ADATA CH11/Memoria - scripts2/sujetos/' + sujeto + '/' + sujeto + '_EyeTracker.pkl')
    eyeTracker_tiempo = pd.read_pickle(path_ + 'unix_et')

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
    
    #baseline
    with open(path + '/baseline/' + sujeto + '_baseline.pkl', 'rb') as f:
        lista_baseline = pickle.load(f)
    
    baseline_eyeT = lista_baseline[5]
    baseline_ppg = lista_baseline[0]['data'].mean()
    baseline_temp = lista_baseline[1]['data'].mean()
    baseline_eeg = lista_baseline[2].drop('time', axis = 1).mean()
    baseline_ecg = lista_baseline[3]['data'].mean()
    baseline_gsr = lista_baseline[4]['conductance'].mean()
    
    if wkl:
        pupila, _ = diametroPupila_(baseline_eyeT)
        baseline_pupila = np.mean(pupila)
    
    #restarles baseline
    ppg_data = ppg['data'] - baseline_ppg
    ppg_peaks = baseline_peaks(ppg['peaks'], baseline_ppg)
    
    temp_data = temp['data']- baseline_temp
    
    ecg_data = ecg['data']- baseline_ecg
    ecg_peaks = baseline_peaks(ecg['peaks'], baseline_ecg)
    
    gsr_con = gsr['conductance'] - baseline_gsr
    
    canales = list(eeg.drop('time', axis = 1).columns)

    eeg_l = []
    for ch in canales:
        eeg_ch = eeg[ch] - baseline_eeg[ch]
        eeg_l.append(eeg_ch)
    eeg_ = pd.DataFrame(eeg_l)
    eeg_ = eeg_.transpose()
    
    #normalizar por zscore - data leak?? 
    ppg_data = z_score(ppg_data) #aplica funcion zscore a cada columna
    temp_data = z_score(temp_data)
    ecg_data = z_score(ecg_data)
    gsr_con = z_score(gsr_con)
    eeg_ = eeg_.apply(z_score)
    
    #arreglar peaks (zscore) -- si peaks distinto de cero, copiar el valor de data con mismo indice
    for i, peaks in enumerate(ppg_peaks):
        if peaks != 0:
            ppg_peaks[i] = ppg_data[i]
        else:
            ppg_peaks[i] = 0
            
    for i, peaks in enumerate(ecg_peaks):
        if peaks != 0:
            ecg_peaks[i] = ecg_data[i]
        else:
            ecg_peaks[i] = 0
            
    #unir df y tiempos
    ppg_ = pd.DataFrame({'data':ppg_data, 'peaks' : ppg_peaks, 'tiempo': ppg['tiempo'], 'tiempo_peaks': ppg['tiempo_peaks']})
    temp_ = pd.DataFrame({'data': temp_data, 'tiempo':temp['tiempo']})
    eeg_['time'] = eeg['time']
    ecg_ = pd.DataFrame({'data':ecg_data, 'peaks' : ecg_peaks, 'tiempo': ecg['tiempo'], 'tiempo_peaks': ecg['tiempo_peaks']})
    gsr_ = {'conductance': gsr_con.values, 'time': gsr['time'].values}
    
    #guardar
    path_guardar = fn.makedir2(path, 'señales_baseline/' + sujeto )
        
    ppg_.to_pickle(path_guardar + 'ppg.pkl')
    temp_.to_pickle(path_guardar + 'temp.pkl')
    eeg_.to_pickle(path_guardar + 'eeg.pkl')
    ecg_.to_pickle(path_guardar + 'ecg.pkl')
    scipy.io.savemat(path_guardar + 'gsr.mat', gsr_)
    
    if wkl:
        if math.isnan(baseline_pupila):
            print('nan pupila baseline')
        else:
            eyeTracker['PupilLeft'] = eyeTracker['PupilLeft'] - baseline_pupila
            eyeTracker['PupilRight'] = eyeTracker['PupilRight'] - baseline_pupila
            eyeTracker.to_pickle(path_guardar + 'eyeTracker.pkl')
    
    
