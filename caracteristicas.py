#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 21:41:29 2017

@author: antonialarranaga
"""
import numpy as np
from scipy.signal import welch, coherence
from scipy import interpolate, integrate
from scipy.stats import stats
import funciones as fn
import matplotlib.pyplot as plt
import funcion_eeg
from biosppy import tools as st
from biosppy import ecg as ecg_
from peakdetect import peakdetect
import pandas as pd
import pywt
import mod_filtros as filtro
from sklearn.preprocessing import StandardScaler

#sacar caracteristicas de señales

#filtro


#ccs generales:
def estadisticas_señales(data):
    """Calcula: promedio, mediana, maximo, varianza, desv std, desviacion de la mediana, kurtosis,
    skewness y moda

    """
    # ensure numpy
    data = np.array(data)
    mean = np.mean(data)# promedio
    median = np.median(data)# mediana
    maxAmp = np.abs(data - mean).max() # amplitud maxima
    var = np.var(data, dtype=np.float64, ddof=1) # varianza #division por n-1(a, dtype=np.float64)
    desvstd = np.std(data, dtype=np.float64, ddof=1) # desv std
    ad = np.sum(np.abs(data - median))# desviacion absoluta 
    kurt = stats.kurtosis(data, bias=False) # kurtosis
    skew = stats.skew(data, bias=False) # skweness
    moda = stats.mode(data)[0][0] #moda
    
    return mean, median, maxAmp, var, desvstd, ad, kurt, skew, moda

 
def get_slope(data, time, show = False): #se hace regresion lineal a los datos, para obtener la tendencia de su pendiente
    time = time - time[0] #sin offset
    slope,intercept, r_value, p_value, std_err = stats.linregress(time, data)

    '''
    r_value is the correlation coefficient and p_value is the p-value for a hypothesis test whose null hypothesis is that the slope is zero.
    
    For more information about correlation you can fin my last post:
    http://glowingpython.blogspot.com/2012/10/visualizing-correlation-matrices.html
    
    And you can find more about p-value here:
    http://en.wikipedia.org/wiki/P-value
    '''
    if show == True:
        plt.figure()
        x = time
        y = slope*x + intercept
        plt.plot(x, data, 'b.')
        plt.plot(x, y, 'r', label = 'regresión lineal')
        plt.legend()
        plt.xlabel('Tiempo [ms]')
        plt.ylabel('Temperatura [C]')
        plt.title('Temperatura Corporal')
        plt.grid()
        plt.show()
    return slope

def get_powerSpect(data, fs):
    if data.size < 3:
        npe = data.size
    else:
        npe = int(data.size/3)
    frequency, power = welch(data, fs, nperseg= npe)
    return frequency, power

def obtener_bandaf(fc1, fc2, power, frequency):
    power_ = []
    freq = []
    fp = np.column_stack([frequency, power])
    #magnitud vs freq
    i1=0
    i2 = 0
    fc1 = fn.takeClosest(frequency, fc1)
    fc2 = fn.takeClosest(frequency, fc2)
    for index, elemento in enumerate(fp):
        if elemento[0] == fc1:
            i1 = index
           
            break
    for index, elemento in enumerate(fp):
        if elemento[0] == fc2:
            i2 = index
            
            break
    power_ = power[i1: i2]
    freq = frequency[i1: i2]
    return power_, freq

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
    
    tiempo = np.array(eyeTracker['Timestamps_UNIX']) #referencial
    #tiempo_ = tiempo - tiempo[0] 
    
    #sacadas como NAN para dp interpolar 
    for index, tipo in enumerate(datosSacadas): 
        if tipo == 'Saccade':
            diametroPupila[index] = float('nan')
            
    #interpolar NANs (esto hacerlo para cada ventana, pq 5 primeros minutos es de relajacion cn ojos cerrados)
    indicesNAN = np.isnan(diametroPupila)
    
    
    pupila_sinNAN = diametroPupila[~indicesNAN]
    tiempo_sinNAN = tiempo[~indicesNAN]
    
    if len(pupila_sinNAN) == 0 or np.abs(len(pupila_sinNAN) - len(diametroPupila)) > len(diametroPupila)*0.5: # - poner un umbral para que sea valida medida
        #print('no existe info pupila ' + str(np.abs(len(pupila_sinNAN) - len(diametroPupila))))
        return [np.nan], 0
    
    
    
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
    
    if show == True:
        pl = plt.figure()
        ax1 = plt.subplot(311)
        ax1.plot(tiempo, diametroPupila, 'o-', markersize=1, label = 'data original') #conNANs
        ax1.grid()
        ax1.legend()
        plt.xlabel('Timestamps [ms]')
        plt.ylabel('Diámetro Pupila [mm]')
        pl.suptitle('Procesamiento señal de diametro pupilar')
        ax2 = plt.subplot(312, sharex = ax1, sharey = ax1)
        ax2.plot(tpo, pupila_interpolada, 'r.-', label = 'interpolada')
        ax2.grid()
        ax2.legend()
        plt.xlabel('Timestamps [ms]')
        plt.ylabel('Diámetro Pupila [mm]')
        ax3 = plt.subplot(313, sharex = ax1, sharey = ax1)
        ax3.plot(tpo, pupila_filtrada, 'c.-', label = 'filtrada')
        ax3.grid()
        ax3.legend()
        plt.xlabel('Timestamps [ms]')
        plt.ylabel('Diámetro Pupila [mm]')
        
        plt.show()
    
    return pupila_filtrada, tpo

def num_sacadasFijaciones(eyeTracker):

    numero_fijaciones = 0
    duracion_fijaciones = 0
    
    indices_eyeT = np.array(eyeTracker.index)
    
    #fijaciones
    indices_fijaciones = np.array(eyeTracker['FixationIndex'])
    indices_duracion = eyeTracker['GazeEventDuration'] #ms
    
    noNan = ~np.isnan(indices_fijaciones)
    indices_fijaciones = indices_fijaciones[noNan]
    indices_eyeTfijaciones = indices_eyeT[noNan]
    fijaciones = pd.DataFrame(data = indices_fijaciones, index = indices_eyeTfijaciones)
    fijaciones = fijaciones.drop_duplicates(keep = 'first')    
        
    numero_fijaciones = fijaciones.size
    duracion_fijaciones = indices_duracion[fijaciones.index]
    
    indices_sacadas = np.array(eyeTracker['SaccadeIndex'])    
    noNan = ~np.isnan(indices_sacadas)
    indices_sacadas = indices_sacadas[noNan]
    indices_eyeTsacadas = indices_eyeT[noNan]
    sacadas = pd.DataFrame(data = indices_sacadas, index = indices_eyeTsacadas)
    sacadas = sacadas.drop_duplicates(keep = 'first')
    
    numero_sacadas = sacadas.size
    duracion_sacadas = indices_duracion[sacadas.index] #ms
    
    return numero_fijaciones, duracion_fijaciones, numero_sacadas, duracion_sacadas
            
    
#ECG


def hrv_ccst(tiempo_peaks, amplitud_peaks): #dependen del tpo de la ventana - ver cuales sirven?
    #ademas se calculan ccs de HRV - revisar con cuales quedarse según papers
    '''
    se obtiene: 
    AVNN
    SDNN
    rMSDD
    pNN50
    VLF
    LF
    HF
    LF_HF
    peak_LF
    peak_VLF
    peak_HF
    IBI
    -------
    
    '''
    if len(amplitud_peaks) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    IBI = [y - x for x,y in zip(tiempo_peaks,tiempo_peaks[1:])] #inter beat interval[ms]
    
    #f_inter = interpolate.interp1d(tiempo_peaks, amplitud_peaks, kind = 'cubic') #tipo de interpolación
    #xx = np.linspace(tiempo_peaks[0], tiempo_peaks[tiempo_peaks.size-1], 5000)
    #yy = f_inter(xx)
    #frequency, power = welch(yy, (xx[1]-xx[0])/1000) #/1000 para que este en [segundos] y freq en [Hz] - power en 
      
    #Caracteristicas de HRV - revisar si es asi el concepto con IBI? o con tiempo_peaks
    dif = [np.abs(y - x) for x,y in zip(IBI,IBI[1:])]
    #a = sum(y - x > 50 for x,y in zip(IBI,IBI[1:]))
    #power_vlf, frequency_vlf = obtener_bandaf_ECG(0.003, 0.04, power, frequency)
    #power_lf, frequency_lf = obtener_bandaf_ECG(0.04, 0.15, power, frequency)
    #power_hf, frequency_hf = obtener_bandaf_ECG(0.15,0.4, power, frequency)
    
    if len(IBI) == 0:
        return np.nan, np.nan, np.nan, np.nan
    if len(IBI) == 1:
        return np.mean(IBI), np.std(IBI), 0, IBI
    
    ##temporales
    AVNN = np.mean(IBI)
    SDNN = np.std(IBI)
    rMSDD = np.sqrt(np.mean(dif))
    #pNN50 = a*100/(len(IBI)) #porcentaje de diferencia entre intervalos adyacentes mayores a 50 ms
    
    '''
    ##frecuencia - se necesitan intervalos mas largos
    VLF= integrate.simps(power_vlf, frequency_vlf) #VLF = desde 0.003 a 0.04
    LF= integrate.simps(power_lf, frequency_lf)#desde 0.04 a 0.15 Hz
    HF= integrate.simps(power_hf, frequency_hf)#desde 0.15 a 0.4 Hz
    LF_HF = LF/HF
    peak_LF = frequency_lf[np.where(power_lf == max(power_lf))[0][0]]#frecuencias del max power
    peak_VLF = frequency_vlf[np.where(power_vlf == max(power_vlf))[0][0]]
    peak_HF = frequency_hf[np.where(power_hf == max(power_hf))[0][0]]
    #hay mas en tesis de portugues
    '''
    
    return AVNN, SDNN, rMSDD, IBI

def get_peaksECG(data, timestamps, fs, show = False): #de biosppy
        # segment
    rpeaks, = ecg_.hamilton_segmenter(signal=data, sampling_rate=fs)

    # correct R-peak locations
    rpeaks, = ecg_.correct_rpeaks(signal=data,
                             rpeaks=rpeaks,
                             sampling_rate=fs,
                             tol=0.05)

    # extract templates
    templates, rpeaks = ecg_.extract_heartbeats(signal=data,
                                           rpeaks=rpeaks,
                                           sampling_rate=fs,
                                           before=0.2,
                                           after=0.4)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=rpeaks,
                                   sampling_rate=fs,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(data)
    T = (length - 1) / fs
    ts = np.linspace(0, T, length, endpoint=False)
    ts_hr = ts[hr_idx]
    
    if show == True:
        MAJOR_LW = 2.5
        MINOR_LW = 1.5
        plt.figure()
    
        ymin = np.min(data)
        ymax = np.max(data)
        alpha = 0.1 * (ymax - ymin)
        ymax += alpha
        ymin -= alpha
    
        plt.plot(ts, data, linewidth=MAJOR_LW, label='ECG')
        plt.vlines(ts[rpeaks], ymin, ymax,
                   color='m',
                   linewidth=MINOR_LW,
                   label='R-peaks')
    
        plt.ylabel('Amplitud')
        plt.xlabel('Tiempo [s]')
        plt.legend()
        plt.grid()
        plt.show()

    return rpeaks, hr, ts, ts_hr, hr_idx

def get_peaksPPG(data, tpo, show = False): #no para ventanas pq se corta
    max_peaks = peakdetect(data, lookahead=35) #devuelve min_peaks en [1]
    indice, amplitud = zip(*max_peaks[0])
    indice = list(indice)
    tiempo_peaks = tpo[indice] 
    amplitud_peaks =  data[indice]
    
    if show == True:
        plt.figure()
        ymin = np.min(data)
        ymax = np.max(data)
        alpha = 0.1 * (ymax - ymin)
        ymax += alpha
        ymin -= alpha
        fs = 120
        length = len(data)
        T = (length - 1) / fs
        ts = np.linspace(0, T, length, endpoint=False)
        
        plt.plot(ts, data, label = 'PPG')
        plt.plot(ts[indice], amplitud_peaks, 'r*', label = 'peaks')
        
        plt.ylabel('Amplitud')
        plt.xlabel('Tiempo [ms]')
        plt.grid()
        plt.legend()
        plt.show()
        
    return amplitud_peaks, tiempo_peaks



#PPG y ECG
def peaks_getHR(tiempo_peaks, show = True): #data ya limpia
    #a ritmo cardiaco
    
    dif_entre_peaks = [(y - x) for x,y in zip(tiempo_peaks,tiempo_peaks[1:])] #en ms
    dif_entre_peaks = np.array(dif_entre_peaks)
    dif_entre_peaks = dif_entre_peaks/(1000*60) #min
    hr = 1/dif_entre_peaks
    indx = np.array(np.nonzero(np.logical_and(hr >= 40, hr <= 200)))

    hr = hr[indx]
    tiempo_peaks = np.array(tiempo_peaks)
    tpo = tiempo_peaks[indx]
    
    if show:
        plt.figure()
        plt.grid()
        plt.plot(np.squeeze(tpo), np.squeeze(hr), 'r.-', label = 'ritmo cardiaco')
        plt.legend()
        plt.xlabel('Timestamps[ms]')
        plt.ylabel('Ritmo cardiaco [bpm]')
        plt.show()

    return hr, tpo
#EEG
def get_bandas(signal, nch, sampling_rate):
    
    out = funcion_eeg.get_power_features2(signal=signal, channel = nch,
                             sampling_rate=sampling_rate,
                             size=0.25,
                             overlap=0.5)
    ts_feat = out['ts']
    theta = out['theta']
    alpha= out['alpha']
    beta = out['beta']
    gamma = out['gamma']
    
    
    return ts_feat, theta, alpha, beta, gamma

def get_bandas_wavelet_(signal, sampling_rate):
        b, a = st.get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=32,
                         sampling_rate=sampling_rate)
        
        filtered, _ = filtro._filter_signal2(b, a, signal=signal, check_phase=True, axis=0)
    
        wp = pywt.WaveletPacket(data=filtered, wavelet='db4', mode='symmetric', maxlevel = 3)
        
        #me interesa el coeficiente 3,1 (4-8) y 3,2 (8-12) = theta y alfa
        theta = wp['aad'].data
        alfa  = wp['ada'].data
        aaa = wp['aaa'].data
        aa = wp['aa'].data
        a = wp['a'].data
        return theta, alfa, aaa, aa, a
    
def get_alfaTheta_EEGW_alphaTheta(channels, nchannels = None):
    theta = []
    alfa =[]
    for canal in channels:
        t, a, _,_,_ = get_bandas_wavelet_(np.array(channels[canal]), 128)
        theta.append(t)
        alfa.append(a)
        
    df_a = pd.DataFrame(alfa).transpose()
    df_t = pd.DataFrame(theta).transpose()
    
        
    if nchannels is not None:
        df_a.columns = nchannels
        df_t.columns = nchannels
        
    return df_t, df_a

def get_aproximacionWaveletPckg(channels, nchannels = None):
    aaa_ = []
    aa_ =[]
    a_ = []
    for canal in channels:
        _, _, aaa,aa,a = get_bandas_wavelet_(np.array(channels[canal]), 128)
        aaa_.append(aaa)
        aa_.append(aa)
        a_.append(a)
        
    df_aaa = pd.DataFrame(aaa_).transpose()
    df_aa = pd.DataFrame(aa_).transpose()
    df_a = pd.DataFrame(a_).transpose()    
        
    if nchannels is not None:
        df_aaa.columns = nchannels
        df_aa.columns = nchannels
        df_a.columns = nchannels

    return df_aaa, df_aa, df_a
        
#eeg PSD con welch
def get_PSD_welch(x, fs):
    if x.size < 10:
        npe = x.size
    else:
        npe = int(x.size/2)
    f, pxx = welch(x, fs= fs, nperseg= npe)
    #beta = [12 - 25] Hz
    beta_pxx, beta_f = obtener_bandaf(12, 25.5, pxx, f)

    #alfa = [8 - 12] Hz
    alfa_pxx, alfa_f = obtener_bandaf(8, 12.5, pxx, f)
    
    #theta = []
    theta_pxx, theta_f = obtener_bandaf(4, 8.5, pxx, f)

    return alfa_pxx, beta_pxx, theta_pxx

def get_PSD_bandas_alfaBeta(canales, nchannels = None, fs = 128):
    alfa = []
    beta = []
    for canal in canales:
        a, b, _ = get_PSD_welch(np.array(canales[canal]), fs)
        alfa.append(a)
        beta.append(b)
    df_a = pd.DataFrame(alfa).transpose() #revisar
    df_b = pd.DataFrame(beta).transpose()

    if nchannels is not None:
        df_a.columns = nchannels
        df_b.columns = nchannels
        
    return df_a, df_b

def get_PSD_bandas_alfaBetaTheta(canales, nchannels = None, fs = 128):
    alfa = []
    beta = []
    theta =[]
    for canal in canales:
        a, b, t = get_PSD_welch(np.array(canales[canal]), fs)
        alfa.append(a)
        beta.append(b)
        theta.append(t)
    df_a = pd.DataFrame(alfa).transpose() #revisar
    df_b = pd.DataFrame(beta).transpose()
    df_t = pd.DataFrame(theta).transpose()
    if nchannels is not None:
        df_a.columns = nchannels
        df_b.columns = nchannels
        df_t.columns = nchannels
    return df_a, df_b, df_t

def get_coherence_banda(señal1, señal2, fc1, fc2, fs = 128):
    if señal1.size < 3:
        npe = señal1.size
    else:
        npe = int(señal1.size/3)
    f, cxy = coherence(señal1, señal2, fs = fs, nperseg= npe)  
    banda_cxy, freq = obtener_bandaf(fc1, fc2 + 0.5, cxy, f)     

    return banda_cxy

#gsr - numero de peaks
def get_numPeaks(peaks):
    return np.count_nonzero(peaks)


def escalar_df(df):
    scaler = StandardScaler()     
    df_ = scaler.fit_transform(df)
    return df_