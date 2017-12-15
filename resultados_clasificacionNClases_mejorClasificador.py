#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:17:52 2017

@author: antonialarranaga
"""
import os
import pandas as pd
import numpy as np
#para mejores ventanas/Clasificadores - % clasificación por clase

t =2

def get_infoClasificador(df, clasificador, nClases):
    df = df.filter(like=clasificador, axis = 1)

    df_c = pd.concat([nClases, df], axis = 1, ignore_index= True)
    l = ['nClases']
    l = l + list(df.columns)
    df_c.columns = l
    
    return df_c

def getPorcentaje_Clases(df_datos, nMaxClases):
    matriz = np.empty(shape = (nMaxClases - 1, len(list(df_datos)) -1))
    num = 0
    for i in range(2, nMaxClases + 1):
        df_acc = df_datos[df_datos.nClases == i]
        if df_acc.empty:
            print('no hay clase ' + str(i))
            num+=1
            continue
        df_acc = df_acc.mean()
        print(str(df_acc))
        print('  ') 
        matriz[num:] =np.array(df_acc[1:])
        num+=1
    df = pd.DataFrame(matriz*100)
    df.columns = list(df_datos.columns[1:])
    return df

##cargar resultados y Nclusters_finales
path = os.path.dirname(os.path.realpath(__file__))
'''
path_resultados = path + '/resultados/' + str(t) + '/'

valencia = pd.read_pickle(path_resultados + 'valencia_clasificadores_eegSeleccion.pkl')

if t == 2: #valencia esta desordenada
    pptes = pd.read_pickle(path_resultados + 'participantes_valenciat2.pkl')
    #concatenar y ordenar indices por persona - quedarse con valencia sin persona
    v = pd.concat([pptes, valencia], axis = 1, ignore_index = True)
    l = ['nombre'] +  list(valencia.columns)
    v.columns = l
    v.sort_values(by = ['nombre'], axis = 0, ascending = True, inplace = True)
    v = v.drop(['nombre'], axis = 1)
    valencia = v
    
arousal = pd.read_pickle(path_resultados + 'arousal_clasificadores_eegSeleccion.pkl')

if t == 5: #hay dos partes
    wkl1 = pd.read_pickle(path_resultados + 'wkl_clasificadores_eegSeleccion_hastaRR.pkl')
    wkl2 = pd.read_pickle(path_resultados + 'wkl_clasificadores_eegSeleccionDesdeRR.pkl')
    
    wkl = pd.concat([wkl1, wkl2], ignore_index= True)
    
    arousal = arousal[:53] #eliminar espacios de gente con una clase
    valencia = valencia[:48]
    
else:
    wkl = pd.read_pickle(path_resultados + 'wkl_clasificadores_eegSeleccion.pkl')
'''
wkl = pd.read_pickle( path + '/resultados/' + 'wkl_eegCcsZarjam.pkl')

path_resultados = path + '/resultados/' + str(t) + '/'

   
#nClases_valencia = pd.read_pickle(path_resultados + 'nClases_valencia.pkl')
#nClases_arousal = pd.read_pickle(path_resultados + 'nClases_arousal.pkl')
nClases_wkl = pd.read_pickle(path_resultados + 'nClases_wkl.pkl') 

#nClases_valencia = nClases_valencia[nClases_valencia['n_final'] != 1]
#nClases_arousal = nClases_arousal[nClases_arousal['n_final'] != 1]
nClases_wkl = nClases_wkl[nClases_wkl['n_final'] != 1]

#nClases_valencia  = nClases_valencia['n_final'].reset_index(drop = True)
#nClases_arousal  = nClases_arousal['n_final'].reset_index(drop = True)
nClases_wkl = nClases_wkl['n_final'].reset_index(drop = True)

'''
if t == 5: #valencia y arousal
    valencia_f1 = get_infoClasificador(valencia, 'knn_3', nClases_valencia)
    valencia_acc = get_infoClasificador(valencia, 'svmRbf_1_', nClases_valencia)
    
    arousal_f1 = get_infoClasificador(arousal, 'svmRbf_10_', nClases_arousal)
    arousal_acc = get_infoClasificador(arousal, 'svmRbf_1_', nClases_arousal)
    
else: #t ==2
    wkl_f1 = get_infoClasificador(wkl, 'svmRbf_10_', nClases_wkl)
    wkl_acc = get_infoClasificador(wkl, 'svmRbf_1_', nClases_wkl)
'''
wkl_f1 = get_infoClasificador(wkl, 'svmRbf_10_', nClases_wkl)

##resultados clasificacion
#nMax_Valencia = max(nClases_valencia)
#nMax_arousal = max(nClases_arousal)
nMax_wkl = max(nClases_wkl)

if t == 5:
    print('\x1b[1;45m VALENCIA f1 \x1b[0m')
    df_vF1 = getPorcentaje_Clases(valencia_f1, nMax_Valencia)
    print('\x1b[1;45m VALENCIA acc \x1b[0m')
    df_vAcc = getPorcentaje_Clases(valencia_acc, nMax_Valencia)
    print('   ')
    print('\x1b[1;45m AROUSAL f1 \x1b[0m')
    df_aF1 = getPorcentaje_Clases(arousal_f1, nMax_arousal)
    print('\x1b[1;45m AROUSAL acc \x1b[0m')
    df_aAcc = getPorcentaje_Clases(arousal_acc, nMax_arousal)

else: #t == 2
    print('\x1b[1;45m WKL f1 \x1b[0m')
    df_wF1 =  getPorcentaje_Clases(wkl_f1, nMax_wkl)
    #print('\x1b[1;45m WKL acc \x1b[0m')
    #df_wAcc = getPorcentaje_Clases(wkl_acc, nMax_wkl)
 