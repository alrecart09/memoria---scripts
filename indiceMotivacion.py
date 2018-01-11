#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 09:35:56 2017

@author: antonialarranaga
"""

#modificacion extraccion de ccs EEG emociones - añadir mas ccs

import os
import pandas as pd
import numpy as np
import caracteristicas as cc
import funciones as fn
import pickle

#import warnings

#warnings.simplefilter("error")
 
path = os.path.dirname(os.path.realpath(__file__))
t =5 #wkl

participantes = fn.listaParticipantes()[0]
#participantes = participantes[24:] #falta8, 24
#participantes =[]
#participantes = ['roberto-rojas', 'juan-zambrano']

ccs_ = ['numFijaciones', 'numSacadas', 'promPupila', 'varPupila', 'promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'pendienteTemp', 'promTemp', 'medianaTemp', 'numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin']
#ccs_wkl_ = ['e_totalF3_theta', 'e_totalF4_theta', 'e_totalF7_theta', 'e_totalF8_theta', 'entropiaNorm_F3_theta', 'entropiaNorm_F4_theta', 'entropiaNorm_F7_theta', 'entropiaNorm_F8_theta', 'stdF3_theta', 'stdF4_theta', 'stdF7_theta', 'stdF8_theta', 'e_totalP7_alfa', 'e_totalP8_alfa', 'entropiaP7_alfa', 'entropiaP8_alfa', 'stdP7_alfa', 'stdP8_alfa']
#ccs_mot_ = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'TC6', 'F4', 'F8', 'AF4']

ccs_mot_ = ['conjunto']
#ccs_valenc_ = ['beta-alfaF3', 'beta-alfaF4', 'beta-alfaF7', 'beta-alfaF8', 'e_totalF3_beta', 'e_totalF4_beta', 'e_totalF7_beta', 'e_totalF8_beta', 'e_totalP7_beta', 'e_totalP8_beta', 'cF7F8', 'asimetria_a/b_F4F3', 'asimetria_a/b_F8/F7']
#ccs_arousal_ = ['e_totalP7_beta', 'e_totalP8_beta', 'cP7O2', 'cP8O1', 'cP7P8', 'cO1O2', 'b/a_AF3', 'b/a_AF4', 'b/a_F3', 'b/a_F4']

#participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


for sujeto in participantes:
    
    path_ccsWkl = fn.makedir2(path, 'indiceMotivacion/' + str(t))
    ccs_ = ['numFijaciones', 'numSacadas', 'promPupila', 'varPupila', 'promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'pendienteTemp', 'promTemp', 'medianaTemp', 'numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin']
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    num = 0 #num ventana
    
    listaVentanas = fn.listaVent(sujeto, '/ventanasU/' + str(t) + '/')
    #listaVentanas = []
    #listaVentanas = ['330.pkl']
    
    path_ventana = path +'/sujetos/'+ sujeto + '/ventanasU/' + str(t) +  '/'    
    path_ccs = fn.makedir(sujeto, path, 'caracteristicas/' + str(t) )
    
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
    
    matriz_ccs = np.empty(shape = (cant_ventanas, len(ccs_))) #shape en f = num_ventanas, c = num_ccs
    matriz_motivacion = np.empty(shape = (cant_ventanas, len(ccs_mot_))) #84
 
    actividades = []
    vent_nulas = 0
    for ventana in listaVentanas:
        with open(path_ventana + ventana, 'rb') as f:
            lista_ventana = pickle.load(f)
           
        ppg = lista_ventana[1]
        temp = lista_ventana[2]
        eeg = lista_ventana[3]
        ecg = lista_ventana[4]
        gsr = lista_ventana[5] #ojo que está en segundos - *1000 para ms
        eyeT = lista_ventana[6]
        
        if any(x == 0 for x in [ppg.size, temp.size, eeg.size, ecg.size, gsr.size, eyeT.size]):
            print("ventana nula = " + ventana)
            vent_nulas += 1
            continue

        actividades.append(lista_ventana[0][0])
        
        lista_caracteristicas = []
        lista_caracteristicas_eeg_wkl = []
        lista_caracteristicas_eeg_valencia = []
        lista_caracteristicas_eeg_arousal = []

        #eeg - bandas de frec
        
        #eeg_data = eeg.drop(['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'P8', 'T8', 'TC6',
        #'F4', 'F8', 'AF4','time'],axis=1)
        eeg_data = eeg.drop('time', axis = 1)
       
        #eeg_emociones = eeg_data
        alpha, beta, theta= cc.get_PSD_bandas_alfaBetaTheta(eeg_data, nchannels =eeg_data.columns, fs = 128) #power
                
        indice_motivacion = []
        
        alfa_ = alpha.sum().sum() #energia total de todos los canal
        beta_ = beta.sum().sum()
        theta_ = theta.sum().sum()
        indice_motivacion = beta_/(alfa_+theta_)
        
        matriz_motivacion[num,:] = indice_motivacion

        #eyeTracker - prom y var diametro pupila, numero-duracion sacadas/fijaciones
        diametro, tpo = cc.diametroPupila_(eyeT, show = False)# - usarlo? poca info en algunas medidas
        numeroFijaciones, duracionFijaciones, numeroSacadas, duracionSacadas = cc.num_sacadasFijaciones(eyeT)
        #print(str(num))
        #if num == 303:
        # break
        
        if np.isnan(diametro[0]): #implica que es nan pq no es un arreglo
            media_pupila = np.nan
            var_pupila = np.nan
        else: #ddof tira warning cndo diametro es nan pq es 1
            media_pupila = np.mean(diametro) #nan si no hay info pupila
            var_pupila = np.var(diametro, dtype=np.float64, ddof=1) #nan si no hay info pupila

        lista_caracteristicas.extend((numeroFijaciones, numeroSacadas, media_pupila, var_pupila))
      
        #ecg - prom, mediana, var ecgmad
        prom_ecg = np.mean(ecg['data'])
        mediana_ecg = np.median(ecg['data'])
        ecgMad = np.abs(ecg['data'] - mediana_ecg)
        var_ecgMad = np.var(np.array(ecgMad), dtype=np.float64, ddof=1)
        
        lista_caracteristicas.extend((prom_ecg, mediana_ecg, var_ecgMad))
        
        #HR - prom, std, rms
        i_peaks = ecg['tiempo_peaks'].nonzero()
        tpo_peaks = np.array(ecg['tiempo_peaks'])
        peaks = np.array(ecg['peaks'])
        tpo_peaks = tpo_peaks[i_peaks]
        hr, ts_hr = cc.peaks_getHR(tpo_peaks, show = False)

        if hr.size == 0:
            prom_hr = np.nan
            std_hr = np.nan
            rms_hr = np.nan
            AVNN = np.nan
            SDNN = np.nan 
            rMSDD = np.nan
        else:
            rms_hr = np.sqrt(np.sum(np.square(hr))/hr.size)
            prom_hr = np.mean(hr)
            std_hr = hr.std() 
            AVNN, SDNN, rMSDD, _ = cc.hrv_ccst(tpo_peaks, peaks[i_peaks])
        lista_caracteristicas.extend((prom_hr, std_hr, rms_hr, AVNN, SDNN, rMSDD))
        
        
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
        
        #ppg - mean, std, mediana, min, max
        ppg_promedio = np.mean(ppg['data'])
        ppg_std = ppg['data'].std()
        ppg_median = np.median(ppg['data'])
        ppg_max = np.max(ppg['data'])
        ppg_min = np.min(ppg['data'])
        
        lista_caracteristicas.extend((ppg_promedio, ppg_std, ppg_median, ppg_max, ppg_min))
        #print(num)
        #if num == 79:
        #    break
        
        matriz_ccs[num] = np.array(lista_caracteristicas)
        
        num+=1
                
    ccs = pd.DataFrame(matriz_ccs, columns = ccs_)
    motivacion = pd.DataFrame(matriz_motivacion, columns = ccs_mot_)
    #ccs_valenc = pd.DataFrame(matriz_eeg_valencia, columns= ccs_valenc_)
    #ccs_arousal = pd.DataFrame(matriz_eeg_arousal, columns = ccs_arousal_)
    
    if vent_nulas:
        print('Se borra espacio ventana nula')
        ccs = ccs[:-vent_nulas]
        motivacion = motivacion[:-vent_nulas]
        #ccs_arousal = ccs_arousal[:-vent_nulas]
        #ccs_valenc = ccs_valenc[:-vent_nulas]
        
    #ccs_wkl = pd.DataFrame(matriz_eeg_wkl)
    #ccs_valenc = pd.DataFrame(matriz_eeg_valencia)
    #ccs_arousal = pd.DataFrame(matriz_eeg_arousal)
    
    #eliminar filas con algun valor nan en HR - uno o dos latidos (ventanas más cortas)
    indices_nullHR = ccs['AVNN'].index[ccs['AVNN'].apply(np.isnan)]
    print('cantidad ventanas = ' + str(len(ccs)) + ' - cantidad ventanas NAN HR = ' + str(len(list(indices_nullHR))))
    
    ccs = ccs.drop(indices_nullHR)
    motivacion =motivacion.drop(indices_nullHR)
    #ccs_arousal = ccs_arousal.drop(indices_nullHR)
    #ccs_valenc = ccs_valenc.drop(indices_nullHR)
    actividades = pd.DataFrame(actividades).drop(indices_nullHR)
    
    #para cada sujeto, si existe nan en diametro pupila - sacar esa caracteristica e imprimir nombre 
    indices_nullPupila = ccs['promPupila'].index[ccs['promPupila'].apply(np.isnan)]
    if (len(ccs) - len(indices_nullPupila)) > len(ccs)*0.8:
        print('Se borran ventanas con pupila nula = ' + str(len(indices_nullPupila)))
        ccs = ccs.drop(indices_nullPupila)
        motivacion = motivacion.drop(indices_nullPupila)
        #ccs_arousal = ccs_arousal.drop(indices_nullPupila)
        #ccs_valenc = ccs_valenc.drop(indices_nullPupila)
        actividades = pd.DataFrame(actividades).drop(indices_nullPupila)
    else:
        print('Se borra caracteristica de pupila prom y varianza - cant ventanas NAN = ' + str(len(indices_nullPupila)))
        ccs = ccs.drop(['promPupila', 'varPupila'], axis = 1)
        ccs_ = ['numFijaciones', 'numSacadas', 'promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'pendienteTemp', 'promTemp', 'medianaTemp', 'numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin']
    
    
    ccs = pd.DataFrame(ccs, columns = ccs_)
    
    motivacion = pd.DataFrame(motivacion, columns = ccs_mot_)
    #ccs_valenc = pd.DataFrame(ccs_valenc, columns= ccs_valenc_)
    #ccs_arousal = pd.DataFrame(ccs_arousal, columns = ccs_arousal_)
    #ccs_wkl = pd.DataFrame(matriz_eeg_wkl)
    #ccs_valenc = pd.DataFrame(matriz_eeg_valencia)
    #ccs_arousal = pd.DataFrame(matriz_eeg_arousal)

    #guardar matriz en pickle ccs_t.pkl, eeg_wkl.pkl, eeg_arousal.pkl, eeg_valencia.pkl, actividades.pkl
    #ccs.to_pickle(path_ccs + 'ccs.pkl')
    motivacion.to_pickle(path_ccsWkl +  sujeto + '_indiceMotivacionCombinado.pkl')
    #ccs_arousal.to_pickle(path_ccs + 'ccs_arousal.pkl')
    #ccs_valenc.to_pickle(path_ccs + 'ccs_valencia.pkl')
    #actividades.to_pickle(path_ccs + 'actividades_ccs.pkl')
    