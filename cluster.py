#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:54:42 2017

@author: antonialarranaga
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#basado en https://stackoverflow.com/questions/37576527/finding-the-point-of-intersection-of-two-line-graphs-drawn-in-matplotlib
def interseccion_lineas(x1, y1, x2, y2):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    
    
    
    #ax.plot(x1, y1, color='lightblue',linewidth=3)
    #ax.plot(x2, y2, color='darkgreen', marker='^')
    
    # Get the common range, from `max(x1[0], x2[0])` to `min(x1[-1], x2[-1])`   
    x_begin = max(x1[0], x2[0])     # 3
    x_end = min(x1[-1], x2[-1])     # 8
    
    points1 = [t for t in zip(x1, y1) if x_begin<=t[0]<=x_end]  # [(3, 50), (4, 120), (5, 55), (6, 240), (7, 50), (8, 25)]
    points2 = [t for t in zip(x2, y2) if x_begin<=t[0]<=x_end]  # [(3, 25), (4, 35), (5, 14), (6, 67), (7, 88), (8, 44)]
    indices = []
    idx_ = []
    idx = 0
    nrof_points = len(points1)
    while idx < nrof_points-1:
        # Iterate over two line segments
        y_min = min(points1[idx][1], points1[idx+1][1]) 
        y_max = max(points1[idx+1][1], points2[idx+1][1]) 
    
        x3 = np.linspace(points1[idx][0], points1[idx+1][0], 1000)      # e.g., (6, 7) intersection range
        y1_new = np.linspace(points1[idx][1], points1[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (240, 50) in y1
        y2_new = np.linspace(points2[idx][1], points2[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (67, 88) in y2
    
        tmp_idx = np.argwhere(np.isclose(y1_new, y2_new, atol=0.1)).reshape(-1)
        if any(tmp_idx):
            #ax.plot(x3[tmp_idx], y2_new[tmp_idx], 'ro')                 # Plot the cross point
            indices.append(x3[tmp_idx])
            idx_.append(y2_new[tmp_idx])
        idx += 1
    
    plt.show()
    return indices, idx_


    
def bootstrap_resample(X, n=None): #de http://nbviewer.jupyter.org/gist/aflaxman/6871948
    """ Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])  # TODO: write a test demonstrating why array() is important
    return X_resample

def mapeo_labels_SOMcluster(bmu, etiquetas):
    labels = []
    for x, y in bmu:
       # print(x,y)
        n = etiquetas[x,y]
       # print(n)
        labels.append(n)
    return labels

#aca hay funciones buenas http://www.turingfinance.com/clustering-countries-real-gdp-growth-part2/