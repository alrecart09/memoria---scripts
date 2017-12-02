#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:41:33 2017

@author: antonialarranaga
"""
import os
import itertools
import funciones as fn
import pandas as pd
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import linalg
import numpy as np
from sklearn import manifold, datasets
from time import time
from matplotlib.ticker import NullFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
import caracteristicas as cc
import matplotlib.pyplot as plt
import cluster as cl
from scipy.cluster import hierarchy
import hdbscan
import seaborn as sns
from sklearn.decomposition import PCA


sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}

###realizar clustering 
path = os.path.dirname(os.path.realpath(__file__))
t = 5
participantes = fn.listaParticipantes()[0]

participantes = []
participantes = ['javier-rojas']


for sujeto in participantes:
    print(sujeto)
    
    path_ccs = path +'/sujetos/'+ sujeto + '/caracteristicas/' + str(t) +  '/' 
    caracteristicas_wkl =  pd.read_pickle(path_ccs + 'ccs_wkl.pkl')
    caracteristicas = pd.read_pickle(path_ccs + 'ccs.pkl')
    valencia = pd.read_pickle(path_ccs + 'ccs_valencia.pkl')
    arousal = pd.read_pickle(path_ccs + 'ccs_arousal.pkl')
    #ccs_ = caracteristicas[['promPupila', 'varPupila']]
    ccs_v = caracteristicas[['gsrAcum', 'promGSR', 'powerGSR', 'maxFasica', 'numPeaksFasica', 'promFasica']]
    ccs_a = caracteristicas[['promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD']]
    
    data = caracteristicas_wkl
    break
#%% bgmm y tsne (visualizacion)
    bgmm = mixture.BayesianGaussianMixture(n_components = 5, covariance_type='full').fit(caracteristicas_wkl)
    
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=0, perplexity=30)
    Y = tsne.fit_transform(caracteristicas_wkl)
    t1 = time()
    print("S-curve, perplexity=%d in %.2g sec" % (5, t1 - t0))
    fig, ax = plt.subplots()
    
    #plt.plot.set_title("Perplexity=%d" % 1000)
    ax.scatter(Y[:, 0], Y[:, 1])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

    #plot_results(caracteristicas_wkl, bgmm.predict(caracteristicas_wkl), bgmm.means_, bgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')
    #http://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html#sphx-glr-auto-examples-mixture-plot-concentration-prior-py
    #http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
    data = np.float32(caracteristicas_wkl)
    n_rows, n_cols = 10, 10
    som = somoclu.Somoclu(n_cols, n_rows, compactsupport = False)
    som.train(data)
    som.view_umatrix()

#%% agglomerative 
    
    knn_graph = kneighbors_graph(data, 30, include_self=False)
    calinski = []
    silueta = []
    for k in range(2,11):
        af = AgglomerativeClustering(n_clusters =k,linkage="ward", connectivity=knn_graph).fit(data)  
        labels = af.labels_
        
        n_clusters_ = np.unique(labels).size
        
        print('Estimated number of clusters: %d' % n_clusters_)
        s = metrics.silhouette_score(data, labels, metric='euclidean')
        silueta.append(s)
        print("Silhouette Coefficient: %0.3f"
              % s )
        ch = metrics.calinski_harabaz_score(data, labels)
        calinski.append(ch)
        print("Calinsky: " +  str(ch))
       
    siluet = cc.escalar_df(pd.DataFrame(silueta))
    calinks = cc.escalar_df(pd.DataFrame(calinski))
    #x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    #plt.plot(x, calinks, '*-', label = 'calinski')
    #plt.plot(x, siluet, '.-', label = 'silueta')

    
    #no tiene sentido intersectar CH con silueta
    #idx, id_ = cl.interseccion_lineas(x, calinks, x, siluet)
    
    #n_clusters = np.round(np.mean(idx[0]))
    #plt.plot(np.mean(idx[0]), np.mean(id_[0]), 'ko', label = 'interseccion real')
    #plt.plot(np.round(np.mean(idx[0])), np.mean(id_[0]), 'ro', label = 'interseccion aproximada')
    
    #plt.legend()
    #plt.grid()
    
    #plt.show()
    #print('numero clusters = ' + str(n_clusters))
    
    #af = AgglomerativeClustering(n_clusters =int(n_clusters),linkage="ward", affinity="euclidean", connectivity=knn_graph).fit(data)  
    #labels = af.labels_
    
    #plt.figure()
    #plt.plot(labels, '*')
    
    #plt.show()
    
#%% jaccard
    for n in range(2):
        
        #resample data con bootstrap
        df_resampled = pd.DataFrame(index=caracteristicas_wkl.index, columns=caracteristicas_wkl.columns)
        for col in caracteristicas_wkl.columns:
            df_resampled[col] = cl.bootstrap_resample(caracteristicas_wkl[col])
        
        #clusterizar data resampleada
        af = AgglomerativeClustering(n_clusters =int(n_clusters),linkage="ward", affinity="euclidean", connectivity=knn_graph).fit(df_resampled)  
        labels_ = af.labels_
        
        #para cada cluster, buscar cluster mas similar y calcular jaccard
    
        j = metrics.jaccard_similarity_score(labels, labels_) #si tiene distintos numreos? 1 o 3?

#%% hdbscan

    clusterer = hdbscan.HDBSCAN(min_cluster_size = 5)
    cluster_labels = clusterer.fit_predict(data)
    plt.figure()
    clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    
    plt.figure()
    plt.plot(cluster_labels, 'r*')
    
#%% som + aglomerativo
    
    n_rows, n_columns =8, 8 #cantidad de neuronas que quiero (cuan densa la zona)
    som = somoclu.Somoclu(n_columns, n_rows, gridtype='hexagonal')
    som.train(data=np.float32(data), epochs=1000)
    bmu = som.bmus #los puntos originales a que neurona están mapeados x,y
#%%cluster som
    calinski =[]
    silueta = []


    for k in range(2, 11):
        af = AgglomerativeClustering(n_clusters = k,linkage="ward")
        som.cluster(algorithm = af)
        state = som.get_surface_state()
        labels = som.clusters #labels para cada neurona
        mapeo = cl.mapeo_labels_SOMcluster(bmu, labels)  
        ch = metrics.calinski_harabaz_score(data, mapeo)
        calinski.append(ch)
        print("Calinsky: " +  str(ch))
        
        s = metrics.silhouette_score(data, mapeo, metric='euclidean')
        silueta.append(s)
        print("Silhouette Coefficient: %0.3f"% s )
    
    best_k_ch = max(enumerate(calinski),key=lambda x: x[1])[0] + 2
    best_k_s= max(enumerate(silueta),key=lambda x: x[1])[0] + 2
    print('Calinksy ' + str(best_k_ch) + ' vs Silueta ' + str(best_k_s))
    af = AgglomerativeClustering(n_clusters =best_k_ch,linkage="ward", affinity="euclidean")
    som.cluster(algorithm = af)    
    som.view_umatrix(bestmatches = True)
    labels_ = som.clusters
    labels_ = cl.mapeo_labels_SOMcluster(bmu, labels_) 
    
#%%
    