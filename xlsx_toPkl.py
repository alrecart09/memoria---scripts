#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:24:00 2018

@author: antonialarranaga
"""
import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))

participantes = ['catalina-astorga', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']
#falta la d la ivania - sobra cata a :c
for sujeto in participantes:
    print(sujeto)
           
    eyeT_baseline = pd.read_excel('/Volumes/ADATA CH11/antonia_baseline/' + sujeto + '.xlsx')
    eyeT_baseline.to_pickle(path + '/antonia_baseline/' + sujeto + '.pkl')