#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 16:10:24 2017

@author: antonialarranaga
"""

import numpy as np
from datetime import datetime
from datetime import timezone
from dateutil import parser
from bisect import bisect_left
import os
import pandas as pd

def toUnix(date, time): #de fecha-hora a unix
    dd = date[3:5]
    mm = date[0:2]
    yyyy = date[6:10]
    date = dd + '-' + mm + '-' + yyyy
    dt = parser.parse(date +' ' + time + '-03:00')
    return int((dt - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()*1000) #ms
    

def takeClosest(myList, myNumber): #retorna numero mas cercano ssi lista esta ordenada
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return after
    else:
       return before
   
def sync_indices(myList, tpo_i, tpo_f): #retorna indice inicial y final mas cercano a tiempo_i, tiempo_f    
    indi = np.where(myList == takeClosest(myList, tpo_i))[0]
    indf = np.where(myList == takeClosest(myList, tpo_f))[0]
    
    if indi.size > 1:
        indi = indi[0]
    if indf.size > 1:
        indf = indf[0]
    
    return int(indi), int(indf)

def cortar_filas_df(indice_i, indice_f, df): #entrega df entre indice_i, indice_f
    df = df.iloc[indice_i:indice_f, :]
    df = df.reset_index()
    del df['index']
    
    return df

def timeStampEyeTracker(date, time): #solo si necesario - se demora caleta
    data = [toUnix(date, t) for t in time]
    tpo = pd.DataFrame(data)
    return tpo
        
def listaParticipantes():
    path = os.path.dirname(os.path.realpath(__file__))
    path_db = path +'/sujetos/'
    participantes = []
    #para todos los participantes
    for root, dirs, files in os.walk(path_db):
        participantes.append(dirs)
        break
    return participantes

def makedir(sujeto, path, carpeta):
    newpath = path + '/sujetos/' + sujeto + '/' + carpeta + '/'
    if not os.path.exists(newpath):
        os.makedirs(path + '/sujetos/' + sujeto + '/' + carpeta + '/')
    return newpath

def makedir2(path, carpeta):
    newpath = path + '/' + carpeta + '/'
    if not os.path.exists(newpath):
        os.makedirs(path + '/' + carpeta + '/')
    return newpath

def listaVent(sujeto, folder): #lista archivos en folder '/nombre/'
    path = os.path.dirname(os.path.realpath(__file__))
    path_ventana = path +'/sujetos/'+ sujeto + folder
    ventanas = []
    #para todos los participantes
    for root, dirs, files in os.walk(path_ventana):
        ventanas.append(files)
        break
    return ventanas[0]
