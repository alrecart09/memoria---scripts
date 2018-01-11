#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 16:46:44 2018

@author: antonialarranaga
"""

#motivacion por etapa

import os
import pandas as pd
import numpy as np
import caracteristicas as cc
import funciones as fn
from sklearn.preprocessing import MinMaxScaler
#import warnings

#warnings.simplefilter("error")
 
def escalar(df):
    scaler = MinMaxScaler()     
    df_ = scaler.fit_transform(df)
    return df_

path = os.path.dirname(os.path.realpath(__file__))
t = 5 

participantes = fn.listaParticipantes()[0]


matriz =  np.empty(shape = (len(participantes), 6)) #shape en f = num_ventanas, c = num_ccs
i=0
for sujeto in participantes:
    
    motivacion = pd.read_pickle(path + '/indiceMotivacion/' + str(t) + '/' + sujeto + '_indiceMotivacionOccipital.pkl')
    actividades = pd.read_pickle(path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/actividades_ccs.pkl')
    
    #motivacion_ = pd.DataFrame(escalar(motivacion))
    #actividades.reset_index(drop = True, inplace=True)
    
    df = pd.concat([actividades, motivacion], axis = 1)
    df.columns = ['act', 'motivacion']
    
    etapas = []    
    #separar por etapas
    borrador = df[df['act'].str.contains("borrador_")]
    intro = df[df['act'].str.contains("introduccion_")]
    tesis = df[df['act'].str.contains("tesis")]
    d1 = df[df['act'].str.contains("desarrollo1_")]
    d2 = df[df['act'].str.contains("desarrollo2_")]
    conclusion = df[df['act'].str.contains("conclusion_")]
    
    motivacion_borrador = borrador['motivacion'].mean()
    
    motivacion_intro = intro['motivacion'].mean()
    
    motivacion_tesis = tesis['motivacion'].mean()
    
    motivacion_d1 = d1['motivacion'].mean()
    
    motivacion_d2 = d2['motivacion'].mean()
    
    motivacion_conclusion = conclusion['motivacion'].mean()
        
    matriz[i,:] = [motivacion_borrador, motivacion_intro, motivacion_tesis, motivacion_d1, motivacion_d2, motivacion_conclusion]
    i+=1
    
    
df_ = pd.DataFrame(matriz)
df_.columns = ['borrador', 'intro', 'tesis', 'd1', 'd2', 'conclusion']

promMotivacion = df_.mean()
desvMotivacion = df_.std()



    