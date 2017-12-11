#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:54:50 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import pandas as pd
#cantidad de clusters en cada etapa de la escritura del ensayo
 
path = os.path.dirname(os.path.realpath(__file__))
t =2
participantes = fn.listaParticipantes()[0]

def crear_df(lista, col):
    df = pd.DataFrame(lista)
    df.columns = col
    df.reset_index(drop=True, inplace = True)
    return df
    
participantes = []
participantes = ['boris-suazo']
participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

path_clusters = path + '/clusters/' + str(t) + '/'

for sujeto in participantes:
    wkl = False
    if any(sujeto in s for s in participantes_wkl):
        print(sujeto)
        wkl = True
        
    path_caracteristicas = path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    actividades = pd.read_pickle(path_caracteristicas + 'actividades_ccs.pkl')
    etiquetas_arousal = pd.read_pickle(path_clusters + sujeto +'_etiquetas-arousalGSR.pkl')
    etiquetas_valencia = pd.read_pickle(path_clusters + sujeto +'_etiquetas-valenciaHR.pkl')
    
    df_val = crear_df(etiquetas_valencia, ['etiquetasVal'])
    df_ar =crear_df(etiquetas_arousal, ['etiquetasAr'])
    
    if wkl:
        path_wkl = path + '/sujetos/' + sujeto + '/'
        etiquetas_wkl = pd.read_pickle(path_wkl + 'etiquetas-wklPupila_' + str(t) + '.pkl')
        df_wkl = crear_df(etiquetas_wkl, ['etiquetasWkl'])
   
    df_act = crear_df(actividades, ['act'])
    
    if wkl:
        df = pd.concat([df_act, df_val, df_ar, df_wkl], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'val', 'ar', 'wkl']
    else:
        df = pd.concat([df_act, df_val, df_ar], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'val', 'ar'] #, 'wkl']
        
    
    borrador = df[df['act'].str.contains("borrador_")]
    intro = df[df['act'].str.contains("introduccion_")]
    d1 = df[df['act'].str.contains("desarrollo1_")]
    d2 = df[df['act'].str.contains("desarrollo2_")]
    conclusion = df[df['act'].str.contains("conclusion_")]
    
    escribiendo = df[df['act'].str.contains("_escribiendo")]
    no_escribiendo = df[df['act'].str.contains("_noescribiendo")]
    revisando = df[df['act'].str.contains("_revisando")]
    
    
        
    
    