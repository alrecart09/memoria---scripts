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
#cantidad de clusters en cada etapa de la escritura del ensayo
 
path = os.path.dirname(os.path.realpath(__file__))


t = 5 #valencia y arousal
#t= 2 #wkl
wkl = False

participantes = fn.listaParticipantes()[0]

def crear_df(lista, col):
    df = pd.DataFrame(lista)
    df.columns = col
    df.reset_index(drop=True, inplace = True)
    return df

#participantes = []
#participantes = ['israfel-salazar']
#participantes = ['alejandro-cuevas']
#participantes_wkl = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

path_clusters = path + '/clusters/' + str(t) + '/'

for sujeto in participantes:
    '''
    wkl = False
    if any(sujeto in s for s in participantes_wkl):
        print(sujeto)
        wkl = True
    '''
    print(sujeto)
    path_ventanas = path + '/sujetos/' + sujeto + '/ventanasU/'
    listaVentanas = fn.listaVent(sujeto, '/ventanasU/' + str(t) + '/')
    
    path_caracteristicas = path + '/sujetos/' + sujeto + '/caracteristicas/' + str(t) + '/'    
    actividades = pd.read_pickle(path_caracteristicas + 'actividades_ccs.pkl')
    etiquetas_arousal = pd.read_pickle(path_clusters + sujeto +'_etiquetas-arousalGSR.pkl')
    etiquetas_valencia = pd.read_pickle(path_clusters + sujeto +'_etiquetas-valenciaHR.pkl')
    ccs = pd.read_pickle(path_caracteristicas + 'ccs.pkl')
    ccs_eeg_ar = pd.read_pickle(path_caracteristicas + 'ccs_arousal.pkl')
    ccs_eeg_val = pd.read_pickle(path_caracteristicas + 'ccs_valencia.pkl')
    
    columnas_eeg_val = list(ccs_eeg_val.columns)
    columnas_eeg_ar = list(ccs_eeg_ar.columns)

    ccs_eeg_val.reset_index(drop = True, inplace = True)
    ccs_eeg_ar.reset_index(drop = True, inplace = True)
    ccs.reset_index(drop = True, inplace = True)
    
    df_val = crear_df(etiquetas_valencia, ['etiquetasVal'])
    df_ar =crear_df(etiquetas_arousal, ['etiquetasAr'])
    
    if wkl:
        path_wkl = path + '/sujetos/' + sujeto + '/'
        etiquetas_wkl = pd.read_pickle(path_wkl + 'etiquetas-wklPupila_' + str(t) + '.pkl')
        df_wkl = crear_df(etiquetas_wkl, ['etiquetasWkl'])
        ccs_eeg_wkl = pd.read_pickle(path_caracteristicas + 'ccs_wkl.pkl')
        ccs_eeg_wkl.reset_index(drop = True, inplace = True)
        columnas_eeg_wkl = list(ccs_eeg_wkl.columns)
        
   
    df_act = crear_df(actividades, ['act'])
    
    c_ar = ['arousal_' + s for s in columnas_eeg_ar]
    c_val = ['valencia_' + s for s in columnas_eeg_val]
    if wkl:
        df = pd.concat([df_act, df_wkl, ccs, ccs_eeg_wkl], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'etiqueta_wkl'] + list(ccs.columns) + columnas_eeg_wkl
    else:
        df = pd.concat([df_act, df_val, df_ar, ccs, ccs_eeg_ar, ccs_eeg_val], ignore_index=True, axis = 1) #df_wkl
        df.columns=['act', 'val', 'ar'] + list(ccs.columns) + c_ar + c_val #, 'wkl']
    
    path_carpeta = fn.makedir2(path, 'hernan/emociones/' + sujeto)
    
    if df.isnull().values.any():
        print(sujeto + 'hay nans en ccs?')
    
    df.to_csv(path_carpeta + 'caracteristicas.csv')
    
    path_ventana = path +'/sujetos/'+ sujeto + '/ventanasU/' + str(t) +  '/'    
    ccs = pd.read_pickle(path_caracteristicas + 'ccs.pkl')
    indices = list(ccs.index.values)
    ventanas_buenas = pd.read_pickle(path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) + '/ventanas.pkl')
    ventanas_buenas = ventanas_buenas.values.tolist()
    vent = 0
    
    for ventana in ventanas_buenas:
        path_carpeta = fn.makedir2(path, 'hernan/emociones/' + sujeto + '/' + str(vent))
        with open(path_ventana + str(ventana[0]), 'rb') as f:
            lista_ventana = pickle.load(f)
           
        ppg = lista_ventana[1]
        temp = lista_ventana[2]
        eeg = lista_ventana[3]
        ecg = lista_ventana[4]
        gsr = lista_ventana[5] #ojo que está en segundos - *1000 para ms
        eyeT = lista_ventana[6]
        if any(x == 0 for x in [ppg.size, temp.size, eeg.size, ecg.size, gsr.size, eyeT.size]):
            print("ventana nula = " + vent)
            break

        ppg.to_csv(path_carpeta + 'ppg.csv')
        temp.to_csv(path_carpeta + 'temp.csv')
        eeg.to_csv(path_carpeta + 'eeg.csv')
        ecg.to_csv(path_carpeta + 'ecg.csv')
        gsr.to_csv(path_carpeta + 'gsr.csv')
        
        vent+=1
      
    '''
    borrador = df[df['act'].str.contains("borrador_")]
    intro = df[df['act'].str.contains("introduccion_")]
    d1 = df[df['act'].str.contains("desarrollo1_")]
    d2 = df[df['act'].str.contains("desarrollo2_")]
    conclusion = df[df['act'].str.contains("conclusion_")]
    
    escribiendo = df[df['act'].str.contains("_escribiendo")]
    no_escribiendo = df[df['act'].str.contains("_noescribiendo")]
    revisando = df[df['act'].str.contains("_revisando")]
    
    pd.to_pickle() 
    '''
    
    