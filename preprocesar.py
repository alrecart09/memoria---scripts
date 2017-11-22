#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 20:08:37 2017

@author: antonialarranaga
"""
import numpy as np
import pandas as pd
from biosppy.signals import ecg
import matplotlib.pyplot as plt
from scipy import interpolate
from biosppy import tools as st
from funcion_eeg import eeg2
from smooth import smooth
from peakdetect import peakdetect

#preprocesamiento
def temp_(path, sujeto, show = False):
    '''
    Downsample señal a 50 [hz] y la filtra pasabajo 10 Hz
    '''
    fs = 120.48 #temp, gsr y ppg
    
    #preprocesar temperatura: downsample y filtrar
    df_temp = pd.read_pickle(path + sujeto + '_syncTemp.pkl')
    
    data = df_temp['Temperature_Skin_Temperature_CAL']
    
    t = df_temp['Temperature_TimestampSync_Unix_CAL']
    
    
    f_inter = interpolate.interp1d(t, data) #tipo de interpolación
    tpo_downsample = np.linspace(t[0], t[t.size-1], int(50*data.size/fs))
    t_downsample = f_inter(tpo_downsample)
    
    
    
    order = int(0.3 * fs)
    t_filtrada, _, _ = st.filter_signal(signal=t_downsample,
                                      ftype='FIR',
                                      band='lowpass',
                                      order=order,
                                      frequency=10,
                                      sampling_rate=fs)
    
    
    if show == True:
        plt.figure()
        plt.suptitle('Procesamiento señal de temperatura corporal')
        ax1 = plt.subplot(211)
        ax1.plot(t/1000 - t[0]/1000, data, 'b-', label='data')
        ax1.plot(tpo_downsample/1000 - t[0]/1000, t_downsample, 'r-', linewidth=2, label='data downsample')
        ax1.grid()
        ax1.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Temperatura [ºC]')
        ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
        ax2.plot(tpo_downsample/1000 - t[0]/1000, t_filtrada,'k-', label = 'data filtrada')
        ax2.grid()
        ax2.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Temperatura [ºC]')
        plt.show()
        
    return t_filtrada, tpo_downsample

def ppg_(path, sujeto, show = True):
    #preprocesar ppg: filtrar
    fs = 120.48 #temp, gsr y ppg
    path_archivo = path + sujeto + '_syncGSRPPG.pkl'
    df_ppg = pd.read_pickle(path_archivo)
    
    data = df_ppg['GSR_PPG_PPG_A13_CAL']
    
    tpo = df_ppg['GSR_PPG_TimestampSync_Unix_CAL']
           
    data_suavizada = smooth(data,window_len=11,window='blackman')
    data_suavizada = data_suavizada[5:len(data_suavizada)-5]
    order = int(0.3 * fs)
    data_filtrada, _, _ = st.filter_signal(signal=data,
                                      ftype='FIR',
                                      band='lowpass',
                                      order=order,
                                      frequency=16,
                                      sampling_rate=fs)
    len_window = 51 #impar
    data_suavizada_filtrada = smooth(data_filtrada,window_len=len_window,window='blackman')
    data_suavizada_filtrada = data_suavizada_filtrada[int((len_window)/2):len(data_suavizada_filtrada)-int((len_window)/2)]
    
    max_peaks = peakdetect(data, lookahead=35) #devuelve min_peaks en [1]
    indice, amplitud = zip(*max_peaks[0])
    indice = list(indice)
    tiempo_peaks = tpo[indice] 
    amplitud_peaks =  data_suavizada_filtrada[indice]
    
    if show == True:
        plt.figure()
        plt.suptitle('Procesamiento señal de PPG')
        ax1 = plt.subplot(211)
        ax1.plot(tpo/1000 - tpo[0]/1000, data, 'b-', label='data')
        ax1.grid()
        ax1.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('PPG')
        ax2 = plt.subplot(212, sharex = ax1, sharey = ax1)
        ax2.plot(tpo/1000 - tpo[0]/1000, data_suavizada_filtrada, 'k-', label = 'data_suavizada')
        ax2.plot(tiempo_peaks/1000 - tpo[0]/1000, amplitud_peaks, 'ro', label = 'peaks')
        ax2.grid()
        ax2.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('PPG')
        plt.show()

    return data_suavizada_filtrada, tpo, amplitud_peaks, tiempo_peaks

def eeg_(path, sujeto, show = False):
    fs = 128.0 #hz
    ncanales = 14 #14 canales de EEG 
    
    #preprocesar temperatura: downsample y filtrar
    df_eeg = pd.read_pickle(path  + sujeto + '_syncEEG.pkl')
    timestamps = df_eeg['nSeqUnixEEG']
    ch_eeg = df_eeg.loc[:, 'AF3':'AF4']
    ch_nombre = list(ch_eeg.columns.values)
    
    filtered, hampel = eeg2(signal=ch_eeg, canales = ncanales, sampling_rate=fs, show=False)
    
    if show == True:
        pl = plt.figure()
        pl.suptitle('Procesamiento señal EEG - canal AF3')
        ax1 = plt.subplot(211)
        ax1.plot(timestamps/1000 - timestamps[0]/1000, df_eeg['AF3'], 'b-', label = 'data canal AF3')
        ax1.grid()
        ax1.legend()
        plt.xlabel('Tiempo [s] ')
        plt.ylabel('Amplitud')
        
        ax2 = plt.subplot(212, sharex = ax1)
        ax1
        ax2.plot(timestamps/1000 - timestamps[0]/1000, hampel[:,0], 'k-', label = 'data filtrada')
        ax2.grid()
        ax2.legend()
        plt.xlabel('Tiempo [s]')
        plt.ylabel('Amplitud')
        plt.show()
    
    #bandas_frecuencia
    #filtered es pasabanda entre 4-40 Hz
    #ts, filtered, hampel, features_ts, theta, alpha, beta, gamma = eeg2(signal=ch_eeg, canales = ncanales, sampling_rate=fs, show=False)
    return hampel, timestamps, ch_nombre


 
def ecg_(path, sujeto, show = False):
    # load raw ECG signal
    df_ecg = pd.read_pickle(path + sujeto + '_syncECG.pkl')
    sig = df_ecg['ECG']
    timestamps = df_ecg['UnixECG']
    # process it and plot
    out = ecg.ecg(signal=sig, sampling_rate=100., show=False)

    rpeaks = out['rpeaks'] #indices donde estan los peaks de la señal
    filtrada = out['filtered']
    
    tiempo_peaks = timestamps[rpeaks] #ms
    amplitud_peaks = filtrada[rpeaks]
    
    tiempo_peaks = tiempo_peaks.reset_index()
    del tiempo_peaks['index']
    tiempo_peaks = tiempo_peaks.T.squeeze()
    
    if show == True:
            
            pl = plt.figure()
            pl.suptitle('Procesamiento señal ECG')
            ax1 = plt.subplot(211)
            ax1.plot(timestamps/1000 - timestamps[0]/1000, sig, 'b-', label='data')
            ax1.grid()
            ax1.legend()
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud')
            ax2 = plt.subplot(212, sharex = ax1)
            ax2.plot(timestamps/1000 -  timestamps[0]/1000, filtrada, 'k-', label = 'data filtrada')
            ax2.plot(tiempo_peaks/1000 - timestamps[0]/1000, amplitud_peaks, 'ro', label = 'peaks')
            ax2.grid()
            ax2.legend()
            plt.xlabel('Tiempo [s]')
            plt.ylabel('Amplitud')
            plt.show()
            
    return filtrada, timestamps, tiempo_peaks, amplitud_peaks
