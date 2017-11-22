#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 18:48:40 2017

@author: antonialarranaga
"""

import scipy.io
import pandas as pd
import numpy as np
import os
import funciones as fn


path = os.path.dirname(os.path.realpath(__file__)) 
path_db = path +'/sujetos/'
#a = os.walk(path_db)

'''
participantes = []

#para todos los participantes
for root, dirs, files in os.walk(path_db):
    participantes.append(dirs)
    break
'''

#participantes = ['camila-socias',  'carlos-navarro', 'catalina-astorga',  'claudio-soto', 'constantino-hernandez', 'diego-gonzalez', 'felipe-silva', 'francisca-asenjo', 'israfel-salazar', 'ivania-valenzuela', 'javier-rojas', 'jenny-miranda', 'melissa-chaperon', 'michelle-fredes', 'nicolas-mellado', 'rodrigo-chi', 'tom-cataldo']
participantes = ['ricardo-ramos']
for sujeto in participantes:
    print(sujeto)

    nombre = '/sujetos/' + sujeto + '/'
    
    eeg = sujeto + '_EEG.mat'
    ecg = sujeto + '_ECG.mat'
    #eyeT= sujeto + '.xlsx'
    gsrppg = sujeto + '_GSR_PPG.csv'
    tmp = sujeto + '_Temperature.csv'
    hr = sujeto + '_HR.csv'
    
    #abrir mat - EEG
    EEG = path + nombre + eeg #mat
    dict_EEG = scipy.io.loadmat(EEG)
    dict_EEG = dict_EEG['data']
    labels = ['nSeqUnixEEG', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'TC6', 'F4', 'F8', 'AF4', 'GYROX', 'GYROY']
    df_EEG = pd.DataFrame(dict_EEG, columns = labels)
    timestamps_EEG = df_EEG['nSeqUnixEEG'].astype(np.int64)#en Unix
    
    #abrir mat - ECG
    ECG =  path + nombre + ecg #mat
    dict_ECG = scipy.io.loadmat(ECG)
    dict_ECG = dict_ECG['data']
    labels = ['UnixECG', 'ECG']
    df_ECG = pd.DataFrame(dict_ECG, columns = labels)
    timestamps_ECG = df_ECG['UnixECG'].astype(np.int64)#en Unix
    
    #abrir xlsx - EyeTracker
    #eyeTracker =  path + nombre + eyeT #xlsx = read_excel
    df_eyeTracker = pd.read_pickle(path + nombre + sujeto + '_EyeTracker.pkl')
    df_eyeTracker = fn.cortar_filas_df(0, 373492, df_eyeTracker)
    timestamps_eyeTracker = df_eyeTracker['LocalTimeStamp'] #en hh:mm:ss.ms(3)
    
    date = df_eyeTracker['RecordingDate'][1] #fecha
    syncTime_eyeTracker_i = fn.toUnix(date, timestamps_eyeTracker[0])#pasarlo a unix para sincronizar
    syncTime_eyeTracker_f= fn.toUnix(date, timestamps_eyeTracker[df_eyeTracker.shape[0]-1])
    
    #abrir csv - Shimmer 
    gsr_ppg = path + nombre + gsrppg
    temp = path + nombre + tmp
    hr_ = path + nombre + hr
    
    df_gsr_ppg = pd.read_csv(gsr_ppg, skiprows=[0, 2]) #se salta el sep = , y la unidad
    df_temp = pd.read_csv(temp, skiprows=[0, 2]) 
    #df_hr = pd.read_csv(hr_, skiprows=[0,2])
    
    fil, col = df_gsr_ppg.shape #exporta una columna extra - pq terminan en coma - sacar
    df_gsr_ppg = df_gsr_ppg.iloc[:, 0:col-1]
    fil, col = df_temp.shape
    df_temp = df_temp.iloc[:, 0:col-1] 
    #fil, col = df_hr.shape
    #df_hr = df_hr.iloc[:, 0:col-1] 
    
    
    timestamps_gsrPPG = df_gsr_ppg['GSR_PPG_TimestampSync_Unix_CAL'] #timestamps_hr = timestamps_gsrPPG 
    timestamps_gsrPPG = timestamps_gsrPPG.astype(np.int64)
    timestamps_temp = df_temp['Temperature_TimestampSync_Unix_CAL']
    timestamps_temp = timestamps_temp.astype(np.int64)
    
    
    #conversiones de http://www.timestampconvert.com/
    
    #Revisar que eyeTracker se puso a medir después (es el mayor) y terminó antes (es el menor) 
    ##si EyeTracker se puso a medir antes => solo ivan, sinc con eeg
    ##si EyeTracker termina después --> dessync con hora de PC: sync segun diferencia
    ok = 1
    
    if any(x > syncTime_eyeTracker_i for x in (timestamps_EEG[0], timestamps_ECG[0], timestamps_gsrPPG[0], timestamps_temp[0])):
        mayor = np.amax([timestamps_EEG[0], timestamps_ECG[0], timestamps_gsrPPG[0], timestamps_temp[0]])
        ok = 0
        #print('revisar inicio de mediciones '+sujeto)
        #sys.exit()
        
    if any(x < syncTime_eyeTracker_f for x in (timestamps_EEG.iloc[-1], timestamps_ECG.iloc[-1], timestamps_gsrPPG.iloc[-1], timestamps_temp.iloc[-1])):  
        ok = 0
        menor = np.amin([timestamps_EEG.iloc[-1], timestamps_ECG.iloc[-1], timestamps_gsrPPG.iloc[-1], timestamps_temp.iloc[-1]])
        
        #print('revisar fin mediciones '+sujeto)
        #sys.exit()
    
    diferencia = syncTime_eyeTracker_f - menor
    print(str(diferencia/1000))
    ok = 1
    if sujeto == 'ivan-zimmermann': #me equivoque midiendo - partio eeg dp
        syncTime_eyeTracker_i = timestamps_EEG[0]
        ii_EEG, if_EEG = fn.sync_indices(timestamps_EEG, syncTime_eyeTracker_i, syncTime_eyeTracker_f) #encontrar datos más cercanos
        ii_ECG, if_ECG = fn.sync_indices(timestamps_ECG, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
        ii_gsrPPG, if_gsrPPG = fn.sync_indices(timestamps_gsrPPG, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
        ii_temp, if_temp = fn.sync_indices(timestamps_temp, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
        sync_EEG = fn.cortar_filas_df(ii_EEG, if_EEG, df_EEG)
        sync_ECG = fn.cortar_filas_df(ii_ECG, if_ECG, df_ECG)
        sync_gsrPPG = fn.cortar_filas_df(ii_gsrPPG, if_gsrPPG, df_gsr_ppg)
        sync_temp = fn.cortar_filas_df(ii_temp, if_temp, df_temp)
        
        sync_ECG.to_pickle(path + nombre + sujeto + '_syncECG.pkl')  # where to save it, usually as a .pkl
        sync_EEG.to_pickle(path + nombre + sujeto + '_syncEEG.pkl')
        sync_gsrPPG.to_pickle(path + nombre + sujeto + '_syncGSRPPG.pkl')
        sync_temp.to_pickle(path + nombre + sujeto + '_syncTemp.pkl')
        df_eyeTracker.to_pickle(path + nombre + sujeto + '_EyeTracker.pkl')
    
 
   
    
    timestamps_EEG = timestamps_EEG + diferencia +2000
    timestamps_ECG = timestamps_ECG+ diferencia +2000
    timestamps_gsrPPG = timestamps_gsrPPG+ diferencia +2000
    timestamps_temp = timestamps_temp+ diferencia +2000
    
    if any(x > syncTime_eyeTracker_i for x in (timestamps_EEG[0], timestamps_ECG[0], timestamps_gsrPPG[0], timestamps_temp[0])):
        mayor = np.amax([timestamps_EEG[0], timestamps_ECG[0], timestamps_gsrPPG[0], timestamps_temp[0]])
        ok = 0
        print('revisar inicio de mediciones '+sujeto)
    #sys.exit()
    
    if any(x < syncTime_eyeTracker_f for x in (timestamps_EEG.iloc[-1], timestamps_ECG.iloc[-1], timestamps_gsrPPG.iloc[-1], timestamps_temp.iloc[-1])):  
        ok = 0
        menor = np.amin([timestamps_EEG.iloc[-1], timestamps_ECG.iloc[-1], timestamps_gsrPPG.iloc[-1], timestamps_temp.iloc[-1]])
        
        print('revisar fin mediciones '+sujeto)
    #sys.exit()
    
    if ok:   
        ii_EEG, if_EEG = fn.sync_indices(timestamps_EEG, syncTime_eyeTracker_i, syncTime_eyeTracker_f) #encontrar datos más cercanos
        ii_ECG, if_ECG = fn.sync_indices(timestamps_ECG, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
        ii_gsrPPG, if_gsrPPG = fn.sync_indices(timestamps_gsrPPG, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
        ii_temp, if_temp = fn.sync_indices(timestamps_temp, syncTime_eyeTracker_i, syncTime_eyeTracker_f)
    
    
        sync_EEG = fn.cortar_filas_df(ii_EEG, if_EEG, df_EEG)
        sync_ECG = fn.cortar_filas_df(ii_ECG, if_ECG, df_ECG)
        sync_gsrPPG = fn.cortar_filas_df(ii_gsrPPG, if_gsrPPG, df_gsr_ppg)
        sync_temp = fn.cortar_filas_df(ii_temp, if_temp, df_temp)
        #sync_HR = fn.cortar_filas_df(ii_gsrPPG, if_gsrPPG, df_hr)
         
        #para separar en tonica y fasica en matlab
        save_path = path + nombre + sujeto + '_GSR.mat'
        save_gsr = mdict={'conductance': sync_gsrPPG['GSR_PPG_GSR_CAL'].values, 'time': sync_gsrPPG['GSR_PPG_TimestampSync_Unix_CAL'].values}
        scipy.io.savemat(save_path,save_gsr)
    
        #guardar señales sincronizadas
        sync_ECG.to_pickle(path + nombre + sujeto + '_syncECG.pkl')  # where to save it, usually as a .pkl
        sync_EEG.to_pickle(path + nombre + sujeto + '_syncEEG.pkl')
        sync_gsrPPG.to_pickle(path + nombre + sujeto + '_syncGSRPPG.pkl')
        sync_temp.to_pickle(path + nombre + sujeto + '_syncTemp.pkl')
        df_eyeTracker.to_pickle(path + nombre + sujeto + '_EyeTracker.pkl')
        
    #df = pd.read_pickle(file_name) para abrir el archivo .pkl como objeto dp

    #fig = plt.figure()
    #plt.plot(sync_EEG[])
    '''
        fig = plt.figure()
        plt.plot(sync_ECG['ECG'], label = 'ECG')
        plt.legend()
        fig = plt.figure()
        plt.plot(sync_gsrPPG['GSR_PPG_GSR_CAL'], label = 'GSR')
        plt.legend()
        fig = plt.figure()
        plt.plot(sync_gsrPPG['GSR_PPG_PPG_A13_CAL'], label = 'PPG')
        plt.legend()
        fig = plt.figure()
        plt.plot(sync_temp['Temperature_Skin_Temperature_CAL'], label = 'temp')
        plt.legend()
    '''
    #df_eyeTracker.to_pickle(path + nombre + sujeto + '_EyeTracker.pkl')
