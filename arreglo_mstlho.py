#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:08:03 2017

@author: antonialarranaga
"""
import pandas as pd
import os
from scipy import interpolate
import numpy as np

path = os.path.dirname(os.path.realpath(__file__)) 

#arreglar fs de 
downsample = ['miguel-sanchez']
upsample = ['tomas-lagos', 'hector-otarola']

for sujeto in upsample:
    path_db = path +'/sujetos/' + sujeto + '/'
    df = pd.read_pickle(path_db + sujeto + '_syncGSRPPG.pkl')
    ppg= df['GSR_PPG_PPG_A13_CAL']
    gsr= df['GSR_PPG_GSR_CAL']
    t = df['GSR_PPG_TimestampSync_Unix_CAL']
    fs = 10
    
    f_inter1 = interpolate.interp1d(t, ppg) #tipo de interpolación
    tpo_ = np.linspace(t[0], t[t.size-1], int(120*ppg.size/fs))
    ppg_= f_inter1(tpo_)
    
    f_inter2 = interpolate.interp1d(t, gsr)
    data_= f_inter2(tpo_)
    
    df_ = pd.DataFrame({'GSR_PPG_TimestampSync_Unix_CAL': tpo_, 'GSR_PPG_GSR_CAL':data_, 'GSR_PPG_PPG_A13_CAL': ppg_})
    
    df_.to_pickle(path_db + sujeto + '_syncGSRPPG.pkl')
