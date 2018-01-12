#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 21:22:13 2018

@author: antonialarranaga
"""

import pandas as pd

#concatenar caracteristicas

path = '/Volumes/ADATA CH11'
t = 2

ccs_ = ['numFijaciones', 'numSacadas', 'promPupila', 'varPupila', 'promECG', 'medianaECG', 'ecgMAD', 'promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD', 'pendienteTemp', 'promTemp', 'medianaTemp', 'numPeaksFasica', 'maxFasica', 'promFasica', 'gsrAcum', 'promGSR', 'powerGSR', 'ppgProm', 'ppgStd', 'ppgMediana', 'ppgMax', 'ppgMin']
ccs_wkl_ = ['e_totalF3_theta', 'e_totalF4_theta', 'e_totalF7_theta', 'e_totalF8_theta', 'entropiaNorm_F3_theta', 'entropiaNorm_F4_theta', 'entropiaNorm_F7_theta', 'entropiaNorm_F8_theta', 'stdF3_theta', 'stdF4_theta', 'stdF7_theta', 'stdF8_theta', 'e_totalP7_alfa', 'e_totalP8_alfa', 'entropiaP7_alfa', 'entropiaP8_alfa', 'stdP7_alfa', 'stdP8_alfa']


participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'catalina-astorga', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']


lista_ccs = []
for sujeto in participantes:
    path_cc = path + '/señales_baseline/' + sujeto + '/caracteristicas/' + str(t) + '/'
    
    ccs = pd.read_pickle(path_cc + 'ccs.pkl')
    ccs_eeg = pd.read_pickle(path_cc + 'ccs_wkl.pkl')
    
    ccs_wkl =  pd.concat([ccs, ccs_eeg], axis=1)
    
    lista_ccs.append(ccs_wkl)
    
ccs_todas = pd.concat(lista_ccs, axis = 0)
ccs_todas.reset_index(drop=True, inplace=True)

ccs_todas.to_pickle(path + '/señales_baseline/ccs_todosWKL.pkl')
    