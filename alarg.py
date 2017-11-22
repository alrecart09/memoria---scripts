#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 18:11:04 2017

@author: antonialarranaga
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from funciones import cortar_filas_df, sync_indices

path = '/Users/antonialarranaga/Desktop/Memoria - scripts/pruebasalargador/'
sin_alarg = 'ppg_sinAlarg.csv'
con_alarg = 'ppgAlargador.csv'

alargado = pd.read_csv(path+con_alarg, skiprows=[0, 2])
corto = pd.read_csv(path+sin_alarg, skiprows=[0, 2])

alarg = alargado['GSR_PPG_PPG_A13_CAL']
cort = corto['GSR_PPG_PPG_A13_CAL']
plt.title('Comparación señal PPG con y sin alargador')
plt.plot(alarg, 'r-',label = 'con alargador')
plt.plot(cort, 'b-', label = 'sin alargador')
plt.xlabel('Muestras')
plt.ylabel('PPG')
plt.grid()
plt.legend()


'''
#test de GSR pie
path = '/Users/antonialarranaga/Desktop/miedo/'
GSRpie = 'pie_electr_f.csv'
GSRmano = 'mano_pia.csv'
gsr = 'pie_scotch_h.csv'

pie = pd.read_csv(path+GSRpie, skiprows=[0, 2])
tpo_inicial = 1504023140000
tpo_final = 1504023282000
indi, indf = sync_indices(pie['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
pie = cortar_filas_df(indi, indf, pie)
pie_gsr = pie['GSR_PPG_GSR_CAL']/np.amax(pie['GSR_PPG_GSR_CAL'])
pie_gsr = pie_gsr - np.mean(pie_gsr)

mano = pd.read_csv(path+GSRmano, skiprows=[0, 2])
tpo_inicial = 1504118165000
tpo_final = 1504118308000
indi, indf = sync_indices(mano['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
mano = cortar_filas_df(indi, indf, mano)
mano_gsr = mano['GSR_PPG_GSR_CAL']/np.amax(mano['GSR_PPG_GSR_CAL'])
mano_gsr = mano_gsr - np.mean(mano_gsr)

manoj = pd.read_csv(path+'mano_j.csv', skiprows=[0, 2])
tpo_inicial = 1504024320000
tpo_final = 1504024464000
indi, indf = sync_indices(manoj['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
manoj = cortar_filas_df(indi, indf, manoj)
mano_j = manoj['GSR_PPG_GSR_CAL']/np.amax(manoj['GSR_PPG_GSR_CAL'])
mano_j = mano_j - np.mean(mano_j)

pie_ = pd.read_csv(path+gsr, skiprows=[0, 2])
fil, col = pie.shape
tpo_inicial = 1504023811000
tpo_final = 1504023952000
indi, indf = sync_indices(pie_['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
pie_ = cortar_filas_df(indi, indf, pie_)
pie__gsr = pie_['GSR_PPG_GSR_CAL']/np.amax(pie_['GSR_PPG_GSR_CAL'])
pie__gsr = pie__gsr - np.mean(pie__gsr)


manoa = pd.read_csv(path + 'mano.csv', skiprows=[0, 2])
manoa = cortar_filas_df(7, manoa.shape[0], manoa)
manoa_gsr = manoa['GSR_PPG_GSR_CAL']/np.amax(manoa['GSR_PPG_GSR_CAL'])
manoa_gsr = manoa_gsr - np.mean(manoa_gsr)


mano_negro = pd.read_csv(path+'negromano.csv', skiprows=[0, 2])
tpo_inicial = 1504296300000
tpo_final =  1504296452000
indi, indf = sync_indices(mano_negro['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
mano_negro = cortar_filas_df(indi, indf, mano_negro)
mano__negro = mano_negro['GSR_PPG_GSR_CAL']/np.amax(mano_negro['GSR_PPG_GSR_CAL'])
mano__negro = mano__negro - np.mean(mano__negro)

mano_vale = pd.read_csv(path+'valemano.csv', skiprows=[0, 2])
tpo_inicial = 1504295646000
tpo_final =  1504295788000
indi, indf = sync_indices(mano_vale['GSR_PPG_Timestamp_Unix_CAL'], tpo_inicial, tpo_final)
mano_vale = cortar_filas_df(indi, indf, mano_vale)
mano__vale = mano_vale['GSR_PPG_GSR_CAL']/np.amax(mano_vale['GSR_PPG_GSR_CAL'])
mano__vale = mano__vale - np.mean(mano__vale)

fig = plt.figure()
plt.plot(pie_gsr, label = 'pie con electrodos f')
plt.plot(mano__vale, label = 'mano v')
#plt.plot(mano__negro, label = 'mano con electrodos')
#plt.plot(mano_j, label = 'mano j')
plt.plot(pie__gsr, label = 'pie scotch h')
#plt.plot(mano_gsr, label = 'mano p')
plt.plot([6023.5, 6023.5], [-0.2, 0.4], 'k-', lw=2) #parte donde se da vuelta la niña
plt.plot([12047, 12047], [-0.2, 0.4], 'k-', lw=2)
plt.legend()
'''