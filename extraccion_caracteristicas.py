#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 07:35:42 2017

@author: antonialarranaga
"""
import os
import pandas as pd
import numpy as np
import caracteristicas as cc
import funciones as fn
import pickle

 
path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

participantes = []
participantes = ['diego-villegas']

for sujeto in participantes:
    print(sujeto)
    num = 0 #num ventana
    
    listaVentanas = fn.listaVent(sujeto, '/ventanasU/' + str(t) + '/')
    listaVentanas = []
    listaVentanas = ['230.pkl', '0.pkl', '100.pkl']
    
    path_ventana = path +'/sujetos/'+ sujeto + '/ventanasU/' + str(t) +  '/'    
    path_ccs = fn.makedir(sujeto, path, 'caracteristicas')
    
    #normalizar GSR
    path_ = path + '/sujetos/' + sujeto + '/'
    gsr_total = pd.read_pickle(path_ + '/GSR/gsrLedalab.pkl') 
    _, GSRpow = cc.get_powerSpect(gsr_total['conductance'], 10)
    cant_ventanas = len(listaVentanas)
    GSR_norm = sum(gsr_total['conductance'])/cant_ventanas
    GSRpow_norm = np.sum(GSRpow)/cant_ventanas
    
    
    with open(path_ventana + str(cant_ventanas -1) + '.pkl', 'rb') as f:
        ultima_ventana = pickle.load(f)
    
    duracion_ultima = len(ultima_ventana[5])/10
    tpo_gsr = duracion_ultima + (cant_ventanas - 1)*t
    
    matriz_ccs = np.empty(shape = (cant_ventanas, 23)) #shape en f = num_ventanas, c = num_ccs

    for ventana in listaVentanas:
        with open(path_ventana + ventana, 'rb') as f:
            lista_ventana = pickle.load(f)
    
        ppg = lista_ventana[1]
        temp = lista_ventana[2]
        eeg = lista_ventana[3]
        ecg = lista_ventana[4]
        gsr = lista_ventana[5] #ojo que está en segundos - *1000 para ms
        eyeT = lista_ventana[6]
        
        lista_caracteristicas = []
        
        #eyeTracker - prom y var diametro pupila, numero-duracion sacadas/fijaciones
        diametro, tpo = cc.diametroPupila(eyeT, show = False)# - usarlo? poca info en algunas medidas
        numeroFijaciones, duracionFijaciones, numeroSacadas, duracionSacadas = cc.num_sacadasFijaciones(eyeT)
       
        media_pupila = np.mean(diametro) #nan si no hay info pupila
        var_pupila = np.var(diametro, dtype=np.float64, ddof=1) #nan si no hay info pupila
    
        lista_caracteristicas.extend((numeroFijaciones, numeroSacadas, media_pupila, var_pupila))
    
        #eeg - bandas de frec
        ###carga cognitiva: alfa, teta
        ###valencia: 
        ###excitacion
        eeg_data = eeg.drop('time',axis=1)
        ts_feat, theta, alpha, beta, gamma = cc.get_bandas(eeg_data, 14, 128) #power
        bandas = ['theta', 'alpha', 'beta', 'gamma']
        ch_nombre = list(eeg.columns)
        
        for banda in bandas: #guarda solo la ultima - sacar lo que se quiere y entregar vector cc de eeg
            signal = np.c_[eval(banda), ts_feat]
            df = pd.DataFrame(signal, columns = ch_nombre)
            
        #ecg - prom, mediana, var ecgmad
        rpeaks, hr, ts, ts_hr, hr_idx = cc.get_peaksECG(np.array(ecg['data']), np.array(ecg['tiempo']), 100, show = False)
        prom_ecg = np.mean(ecg['data'])
        mediana_ecg = np.median(ecg['data'])
        ecgMad = np.abs(ecg['data'] - mediana_ecg)
        var_ecgMad = np.var(np.array(ecgMad), dtype=np.float64, ddof=1)
        
        lista_caracteristicas.extend((prom_ecg, mediana_ecg, var_ecgMad))
        #HR - prom, std, rms
        prom_hr = np.mean(hr)
        std_hr = np.std(hr, dtype=np.float64, ddof=1)
        rms_hr = np.sqrt(np.sum(np.square(hr))/hr.size)
        AVNN, SDNN, rMSDD, pNN50, _ = cc.hrv_ccst(ts_hr, hr)
        
        lista_caracteristicas.extend((prom_hr, std_hr, rms_hr, AVNN, SDNN, rMSDD, pNN50))
        #ppg - ?
        peaks, tpo = cc.get_peaksPPG(np.array(ppg['data']), np.array(ppg['tiempo']), show = False)
        
        #temp - slope, prom y mediana
        pendiente_temp = cc.get_slope(np.array(temp['data']), np.array(temp['tiempo']), show = False)     
        prom_temp = np.mean(temp['data'])
        mediana_temp = np.median(temp['data'])
        
        lista_caracteristicas.extend((pendiente_temp, prom_temp, mediana_temp))
        
        #comp. fasica - cantidades de peaks, media y abs max
        num_peaks = cc.get_numPeaks(np.array(gsr['peaks_fasica']))
        max_fasica = np.abs(np.amax(gsr['fasica']))
        prom_fasica = np.mean(gsr['fasica'])
        
        lista_caracteristicas.extend((num_peaks, max_fasica, prom_fasica))
        
        #gsr - gsr_acumulado, gsr_avg, power_gsr_norm
        gsr_normalizado =gsr['conductance']/GSR_norm
        gsr_acumulado = np.sum(gsr_normalizado)/tpo_gsr
        gsr_avg = gsr_acumulado/tpo_gsr
        freq, power = cc.get_powerSpect(gsr['conductance'], 10)
        power_gsr_norm = np.mean(power)/GSRpow_norm
        
        lista_caracteristicas.extend((gsr_acumulado, gsr_avg, power_gsr_norm))
        
        matriz_ccs[num] = np.array(lista_caracteristicas)
        num+=1
        
        #guardar matriz en pickle ccs_t.pkl, eeg.pkl
    

    
    