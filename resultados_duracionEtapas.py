#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:54:50 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import pandas as pd
import pickle
import numpy as np
#cantidad de clusters en cada etapa de la escritura del ensayo
 
path = os.path.dirname(os.path.realpath(__file__))


t = 5 #valencia y arousal
#t= 2 #wkl
#wkl = True

participantes = fn.listaParticipantes()[0]

def crear_df(lista, col):
    df = pd.DataFrame(lista)
    df.columns = col
    df.reset_index(drop=True, inplace = True)
    return df

#participantes = []
#participantes = ['israfel-salazar']
#participantes = ['alejandro-cuevas']
participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

path_clusters = path + '/clusters/' + str(t) + '/'
i = 0
clmns = ['borrador_duracion', 'intro_duracion', 'd1_duracion', 'd2_duracion', 'conclusion_duracion','escribiendo_duracion', 'noescr_duracion', 'revisando_duracion']
matriz = np.empty(shape = (len(participantes), len(clmns)))

for sujeto in participantes:
    lista= []
    wkl = False
    if any(sujeto in s for s in participantes_wkl):
        print(sujeto)
        wkl = True
    
    print(sujeto)
    path_ventanas = path + '/sujetos/' + sujeto + '/ventanasU/'
    listaVentanas = fn.listaVent(sujeto, '/ventanasU/' + str(t) + '/')
    
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
        df = pd.concat([df_act, df_wkl,], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'etiqueta_wkl'] 
    else:
        df = pd.concat([df_act, df_val, df_ar, ], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'val', 'ar']
        
    path_ventana = path +'/sujetos/'+ sujeto + '/ventanasU/' + str(t) +  '/'    
    
    ventanas_buenas = pd.read_pickle(path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) + '/ventanas.pkl')
    ventanas_buenas = ventanas_buenas.values.tolist()
    vent = 0
    duracion_ventana = []
    '''
    for ventana in ventanas_buenas:
        with open(path_ventana + str(ventana[0]), 'rb') as f:
            lista_ventana = pickle.load(f)
        temp = lista_ventana[2]
        duracion_ventana.append(len(temp)/50) #[s]
        vent+=1
      
    duracion_vent = crear_df(duracion_ventana, ['seg'])
    borrador = df[df['act'].str.contains("borrador_")]
    borrador_duracion = duracion_vent[df['act'].str.contains("borrador_")].sum()
    intro = df[df['act'].str.contains("introduccion_")]
    intro_duracion = duracion_vent[df['act'].str.contains("introduccion_")].sum()
    d1 = df[df['act'].str.contains("desarrollo1_")]
    d1_duracion = duracion_vent[df['act'].str.contains("desarrollo1_")].sum()
    d2 = df[df['act'].str.contains("desarrollo2_")]
    d2_duracion = duracion_vent[df['act'].str.contains("desarrollo2_")].sum()
    conclusion = df[df['act'].str.contains("conclusion_")]
    conclusion_duracion = duracion_vent[df['act'].str.contains("conclusion_")].sum()
    
    
    escribiendo = df[df['act'].str.contains("_escribiendo")]
    escribiendo_duracion = duracion_vent[df['act'].str.contains("_escribiendo")].sum()
    no_escribiendo = df[df['act'].str.contains("_noescribiendo")]
    noescr_duracion = duracion_vent[df['act'].str.contains("_noescribiendo")].sum()
    revisando = df[df['act'].str.contains("_revisando")]
    revisando_duracion =  duracion_vent[df['act'].str.contains("_revisando")].sum()
    
    lista = [borrador_duracion, intro_duracion, d1_duracion, d2_duracion, conclusion_duracion,escribiendo_duracion, noescr_duracion, revisando_duracion]
    matriz[i,:] = lista
    i+=1
    #pd.to_pickle() 

df_tiempos = pd.DataFrame(data = matriz, columns = clmns)

cant_nans = df_tiempos.isnull().sum()
suma = df_tiempos.sum(skipna = True)

borrador = df_tiempos['borrador_duracion'].dropna()
revisando = df_tiempos['revisando_duracion'].dropna()

resto = df_tiempos.drop(['borrador_duracion', 'revisando_duracion'], axis = 1)

resto.mean()
resto.std()
'''

    