#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:00:50 2017

@author: antonialarranaga
"""
import os
import pandas as pd
#resultado por sensor
    

#cargar resultados
path = path = os.path.dirname(os.path.realpath(__file__))
path_resultados = path + '/resultados/'

valencia = pd.read_pickle(path_resultados + 'valencia_clasificadores_porSensor.pkl')
arousal = pd.read_pickle(path_resultados + 'arousal_clasificadores_porSensor.pkl')
wkl = pd.read_pickle(path_resultados + 'wkl_clasificadores_porSensor.pkl')

valencia = valencia[0:48]
arousal = arousal[0:52]

val_prom = valencia.mean()
ar_prom = arousal.mean()
wkl_prom = wkl.mean()

