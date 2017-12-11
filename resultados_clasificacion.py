#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:14:57 2017

@author: antonialarranaga
"""
import os
import pandas as pd

def get_clasificador(df_prom):
    df_svm = df_prom.filter(like='svm')
    df_ann = df_prom.filter(like = 'ann_')
    df_knn = df_prom.filter(like = 'knn_')
    
    return df_knn, df_svm, df_ann

def get_maximo(df_knn, df_svm, df_ann, criterio):
    #df son promedios de df original
    max_knn_valor = max(df_knn.filter(like = criterio, axis=0))
    max_knn_clas = df_knn.filter(like = criterio, axis=0).argmax()
    
    max_svmV = max(df_svm.filter(like = criterio, axis=0))
    max_svm_clas = df_svm.filter(like = criterio, axis=0).argmax()
    
    max_annV = max(df_ann.filter(like = criterio, axis=0))
    max_ann_clas = df_ann.filter(like = criterio, axis=0).argmax()
    
    max_knn = {'clasificador':max_knn_clas, 'valor_métrica': max_knn_valor}
    max_svm = {'clasificador':max_svm_clas, 'valor_métrica': max_svmV}
    max_ann = {'clasificador':max_ann_clas, 'valor_métrica': max_annV}
    return max_knn, max_svm, max_ann

def get_metricas_Mejorclasificador(df, clasificador):
    df = df.filter(like=clasificador)
    print(str(df))
    return df
#como le meto lo multiclase??
    
#resultados clasificacion
t =2

#cargar resultados
path = path = os.path.dirname(os.path.realpath(__file__))
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

val_prom = valencia.mean()
ar_prom = arousal.mean()
wkl_prom = wkl.mean()

val_knn, val_svm, val_ann = get_clasificador(val_prom)
ar_knn, ar_svm, ar_ann = get_clasificador(ar_prom)
wkl_knn, wkl_svm, wkl_ann = get_clasificador(wkl_prom)

#en torno a que maximizo? f1, acc?
max_f1_val_knn, max_f1_val_svm, max_f1_val_ann = get_maximo(val_knn, val_svm, val_ann, '_f1')
max_f1_ar_knn, max_f1_ar_svm, max_f1_ar_ann = get_maximo(ar_knn, ar_svm, ar_ann, '_f1')
max_f1_wkl_knn, max_f1_wkl_svm, max_f1_wkl_ann = get_maximo(wkl_knn, wkl_svm, wkl_ann, '_f1')

max_acc_val_knn, max_acc_val_svm, max_acc_val_ann = get_maximo(val_knn, val_svm, val_ann, '_acc')
max_acc_ar_knn, max_acc_ar_svm, max_acc_ar_ann = get_maximo(ar_knn, ar_svm, ar_ann, '_acc')
max_acc_wkl_knn, max_acc_wkl_svm, max_acc_wkl_ann = get_maximo(wkl_knn, wkl_svm, wkl_ann, '_acc')

print('VALENCIA t = ' + str(t))
print( 'mejor knn ACC \n' + str(max_acc_val_knn))
print( 'mejor svm ACC \n' + str(max_acc_val_svm))
print( 'mejor ann ACC \n' + str(max_acc_val_ann))
print(  )
print( 'mejor knn f1 \n' + str(max_f1_val_knn))
print( 'mejor svm f1 \n' + str(max_f1_val_svm))
print( 'mejor ann f1 \n' + str(max_f1_val_ann))

print('AROUSAL t = ' + str(t) + '\n')
print( 'mejor knn ACC \n' + str(max_acc_ar_knn))
print( 'mejor svm ACC \n' + str(max_acc_ar_svm))
print( 'mejor ann ACC \n' + str(max_acc_ar_ann))
print(  )
print( 'mejor knn f1 \n' + str(max_f1_ar_knn))
print( 'mejor svm f1 \n' + str(max_f1_ar_svm))
print( 'mejor ann f1 \n' + str(max_f1_ar_ann))

print('WKL t = ' + str(t) + '\n')
print( 'mejor knn ACC \n' + str(max_acc_wkl_knn))
print( 'mejor svm ACC \n' + str(max_acc_wkl_svm))
print( 'mejor ann ACC \n' + str(max_acc_wkl_ann))
print(  )
print( 'mejor knn f1 \n' + str(max_f1_wkl_knn))
print( 'mejor svm f1 \n' + str(max_f1_wkl_svm))
print( 'mejor ann f1 \n' + str(max_f1_wkl_ann))