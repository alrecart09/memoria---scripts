#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:27:39 2017

@author: antonialarranaga
"""
import os
import funciones as fn
import collections
import pandas as pd
import numpy as np
#resultados cluster - matriz que indica cantidad de clusters por persona


path = os.path.dirname(os.path.realpath(__file__))

t = 2
#participantes = fn.listaParticipantes()[0]
participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


clases_original = []
clases_final = []
for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')
    path_ccs = path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) +  '/' 
    caracteristicas = pd.read_pickle(path_ccs + 'ccs.pkl')
    
    pupila = caracteristicas['promPupila']
    #path_etiquetas = path +'/clusters/'+ str(t) + '/'
    path_etiquetas = path + '/sujetos/' + sujeto + '/'
    #etiquetas = pd.read_pickle(path_etiquetas + sujeto + '_etiquetas-arousalGSR.pkl')
    etiquetas = pd.read_pickle(path_etiquetas + 'etiquetas-wklPupila_' + str(t) + '.pkl')
    
    pupila.reset_index(inplace=True, drop = True)
    etiquetas.reset_index(inplace = True, drop = True)
    
    pupEt = pd.concat([etiquetas, pupila], axis = 1)
    pupEt.columns = ['etiquetas', 'promPupila']
    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    print('numero original de clases: ' + str(len(cuenta)), 'd ' + str(repeticion))
    clases_original.append(len(cuenta))
    for clase, cantidad in repeticion:
       # print(str(clase) + ' ' + str(cantidad))
        if cantidad < 6:
            #print('clase ' + str(clase) + ' con <= de 5 elementos')
            
            indices = np.where(pupEt['etiquetas'] == clase)[0]
            #print(indices)
            pupEt = pupEt.drop(list(indices))

            pupEt.reset_index(drop = True, inplace = True)
            #print('borre clase ' + str(clase))
    final =  collections.Counter(pupEt['etiquetas'].values.ravel())
    print('numero final de clases: ' + str(len(final)))
    clases_final.append(len(final))
    

    pupila_clase =[]
    for clase, cantidad in final.most_common():
        ind = np.where(pupEt['etiquetas'] == clase)[0]
        pup = pupEt['promPupila'][ind]
        lista = [clase, pup.mean()]
        pupila_clase.append(lista)
        
    print(pupila_clase)
        
'''   
df = pd.DataFrame({'n_original': clases_original, 'n_final': clases_final})
path_resultados = path + '/resultados/' + str(t) + '/'
df.to_pickle(path_resultados + 'nClases_wkl.pkl')
'''