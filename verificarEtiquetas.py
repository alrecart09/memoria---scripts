#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:11:56 2017

@author: antonialarranaga
"""
'''
verificar etiquetas eyeTracker

'''

import os
import pandas as pd


path = os.path.dirname(os.path.realpath(__file__)) 
path_db = path +'/conEyeTracker/'
a = os.walk(path_db)

participantes = []
#mal_etiquetado = {}

#para todos los participantes
for root, dirs, files in os.walk(path_db):
    participantes.append(dirs)
    break

etiquetas = ['cuestionario_i', 'cuestionario_f', 'instr_i', 'instr_f', 'instr_ens_i', 'borrador_escribiendo',
    'borrador_noescribiendo', 'borrador_revisando', 'introduccion_escribiendo', 
    'relajacion_i', 'relajacion_f', 'tesis_escribiendo', 'tesis_noescribiendo', 
    'tesis_revisando', 'nn', 'introduccion_noescribiendo', 'introduccion_revisando', 'desarrollo1_escribiendo',
    'desarrollo1_noescribiendo', 'desarrollo1_revisando', 'desarrollo2_escribiendo',
    'desarrollo2_noescribiendo', 'desarrollo2_revisando', 'conclusion_escribiendo', 
    'conclusion_noescribiendo', 'conclusion_revisando', 'enviar_i', 'enviar_f']

for participante in participantes[0]:
    print (participante)
    df_eyeTracker = pd.read_excel(path_db + '/' + participante + '/' + participante + '.xlsx', parse_cols = "AL, AK, AJ") #lento
    
    indices = df_eyeTracker['StudioEventIndex']
    etiquetas_evento = df_eyeTracker['StudioEventData'].dropna() #sin nans
    tipo_evento = df_eyeTracker['StudioEvent'].dropna()

    indices_eventos = []

    #lista de etiquetas que son default en tipo_evento (los que yo anoté)
    indices_tipo = tipo_evento.index.values
    indices_etiquetas = etiquetas_evento.index.values


    for i in range(0, tipo_evento.shape[0]):
        if tipo_evento[indices_tipo[i]] == 'Default':
            indices_eventos.append(indices_tipo[i])
        
    etiquetas_evento = etiquetas_evento[indices_eventos] #etiquetas solo de eventos que yo anote

    a = set(etiquetas).symmetric_difference(set(etiquetas_evento))
    if a:
        print('existe diferencia en ' + participante + ' en etiqueta(s) ' +  str(a))

