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
from scipy.stats import stats
import collections
import operator

###realizar clustering 
#path = os.path.dirname(os.path.realpath(__file__))
path = '/Volumes/ADATA CH11'

t =2
participantes = fn.listaParticipantes()[0]



#participantes = []
#participantes = ['israfel-salazar']
#participantes = ['manuela-diaz', 'camila-socias', 'boris-suazo']
#participantes = ['manuela-diaz']
#participantes = ['alejandro-cuevas', 'camila-socias', 'emilio-urbano', 'felipe-silva', 'francisca-barrera', 'israfel-salazar', 'ivan-zimmermann', 'ivania-valenzuela', 'jaime-aranda', 'juan-zambrano', 'manuela-diaz', 'michelle-fredes', 'miguel-sanchez', 'ricardo-ramos', 'roberto-rojas', 'rodrigo-chi']

path_clusters = fn.makedir2(path, 'clusters_todosWKL/' + str(t) )

participantes = ['todos']

for sujeto in participantes:
    print(sujeto)
    
    path_ccs = path +'/señales_baseline/ccs_todosWKL.pkl'
    #caracteristicas_wkl =  pd.read_pickle(path_ccs +  'ccs_wkl_' + str(t) + '.pkl')
    caracteristicas = pd.read_pickle(path_ccs)
    #valencia = pd.read_pickle(path_ccs + 'ccs_valencia.pkl')
    #arousal = pd.read_pickle(path_ccs + 'ccs_arousal.pkl')
    ccs_ = caracteristicas[['promPupila', 'varPupila']]
    #ccs_a = caracteristicas[['gsrAcum', 'promGSR', 'powerGSR', 'maxFasica', 'numPeaksFasica', 'promFasica']]
    #ccs_v = caracteristicas[['promHR', 'stdHR', 'rmsHR', 'AVNN', 'SDNN', 'rMSDD']]
    
    
    data = ccs_
    
    data = cc.escalar_df(data)
    #ccs_wkl = cc.escalar_df(ccs_wkl)
    #ccs_arousal = cc.escalar_df(ccs_arousal)
    #ccs_valenc = cc.escalar_df(ccs_valenc)
    
    # som + aglomerativo
  
    k_best = [] #guardar el mejor labels de cada i
    etiquetas_best = []
    ch__= []
    for i in range(50):
        print(i)
        n_rows, n_columns =6,6 #cantidad de neuronas que quiero (cuan densa la zona)
        som = somoclu.Somoclu(n_columns, n_rows, gridtype='hexagonal')
        som.train(data=np.float32(data), epochs=1000)
        bmu = som.bmus #los puntos originales a que neurona están mapeados x,y
        
        
        calinski =[]
        silueta = []
        etiquetas = []
        for k in range(2,7):
            af = AgglomerativeClustering(n_clusters = k,linkage="average", affinity='cityblock')
            som.cluster(algorithm = af)
            #state = som.get_surface_state()
            labels = som.clusters #labels para cada neurona
            mapeo = cl.mapeo_labels_SOMcluster(bmu, labels)  
            
            number_labels = list(set(mapeo))
            
            if len(number_labels) == 1:
                print('no hay')
                continue
            
            ch = metrics.calinski_harabaz_score(data, mapeo)
            calinski.append(ch)
           # print("Calinsky: " +  str(ch))
            
            s = metrics.silhouette_score(data, mapeo, metric='cityblock')
            silueta.append(s)
           # print("Silhouette Coefficient: %0.3f"% s )
            etiquetas.append(mapeo)
        
        best_k_ch = max(enumerate(calinski),key=lambda x: x[1])[0] + 2
        best_k_s= max(enumerate(silueta),key=lambda x: x[1])[0] + 2
        calinskii = max(calinski)

        #print('Calinksy ' + str(best_k_ch) + ' vs Silueta ' + str(best_k_s))
        
        if best_k_ch == best_k_s:
            #print('me repito en ' + str(best_k_ch) + 'en ' + str(i))
            k_best.append(best_k_ch)
            etiquetas_best.append(etiquetas[best_k_ch-2])
            ch__.append(calinskii)
        
        
        #plt.figure()
        #plt.title("CH=%0.3f" % ch)
        #plt.plot(labels_, '*')
                
    moda = stats.mode(k_best)
    ch_ = []
    lb_ = []
    distr = []
    maxim = []
    
    if len(list(moda)[0]) == 0:
        print('no se repite')
        
    elif moda[1][0] > 0: #si hay un valor q se repite - se elige la moda y el q sea más pequeño
        k_elegido = moda[0][0]
        for idx, num in enumerate(k_best):
            if num == k_elegido:
                #print('hola')
                ch_.append(ch__[idx])
                lb_.append(etiquetas_best[idx])
                y = collections.Counter(etiquetas_best[idx])
                distr.append(y)
                #print(str(y))
        for counter in distr:
            orden = counter.most_common()
            mas_comun = list(orden[0])[1]
            #print(str(mas_comun))
            maxim.append(mas_comun)
            
        indice_elegido, minimo_valor = min(enumerate(maxim), key=operator.itemgetter(1))
        print(indice_elegido)
        #indice_elegido = max(enumerate(ch_),key=lambda x: x[1])[0]
        labels_elegidas = lb_[indice_elegido]

    else: #si ninguno se repite
        print('no pasa ná')
    y = collections.Counter(labels_elegidas)
    print('numero de clusters = ' + str(k_elegido) + ' reparticion ' + str(y))
    
    
    #guardar etiquetas WKL
    etiquetas = pd.DataFrame(labels_elegidas)
    etiquetas.to_pickle(path_clusters + sujeto + 'clusters.pkl')
    
#%%
    '''   
    #graficos 2D pupila
    fig, ax = plt.subplots()
     
    labels_ = labels_elegidas
     
    a = np.where(np.array(labels_) == 0)[0]
    b =  np.where(np.array(labels_) == 1)[0]   
    c=  np.where(np.array(labels_) == 2)[0]
    d = np.where(np.array(labels_) == 3)[0]
    e =  np.where(np.array(labels_) == 4)[0]
    f =  np.where(np.array(labels_) == 5)[0]
    g = np.where(np.array(labels_) == 6)[0]
    h =  np.where(np.array(labels_) == 7)[0]
     
    #i =  np.where(np.array(labels_) == 8)[0]    
    #j =  np.where(np.array(labels_) == 9)[0]
     
     
    data = np.array(data)
    ax.scatter(data[a, 0], data[a, 1], c="r")
    ax.scatter(data[b, 0], data[b, 1], c="g")  
    ax.scatter(data[c, 0], data[c, 1], c="b")
    ax.scatter(data[d, 0], data[d, 1], c="c")
    ax.scatter(data[e, 0], data[e, 1], c="m")        
    ax.scatter(data[f, 0], data[f, 1], c="y")    
    ax.scatter(data[g, 0], data[g, 1], c="k")
    ax.scatter(data[h, 0], data[h, 1], c="k", marker = '^')
     
     #ax.scatter(data[i, 0], data[i, 1], c="r", marker = 'x')
     #ax.scatter(data[j, 0], data[j, 1], c="b", marker = 's')
    ax.grid()   
    
    '''    
    '''    
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='random',
                          random_state=0, perplexity=30, n_iter=5000)
    Y = tsne.fit_transform(data)
    t1 = time()
    print("S-curve, perplexity=%d in %.2g sec" % (5, t1 - t0))
    fig, ax = plt.subplots()
     
     #plt.plot.set_title("Perplexity=%d" % 1000)
    labels_ = labels_elegidas
    a = np.where(np.array(labels_) == 0)[0]
    b =  np.where(np.array(labels_) == 1)[0]
    
    c=  np.where(np.array(labels_) == 2)[0]

    d = np.where(np.array(labels_) == 3)[0]
    e =  np.where(np.array(labels_) == 4)[0]
    f =  np.where(np.array(labels_) == 5)[0]
    g = np.where(np.array(labels_) == 6)[0]
         
    h =  np.where(np.array(labels_) == 7)[0]
    i =  np.where(np.array(labels_) == 8)[0]
     
     
    j =  np.where(np.array(labels_) == 9)[0]
 
  
    #ax = subplots[0][0]
    ax.scatter(Y[a, 0], Y[a, 1], c="r")
    ax.scatter(Y[b, 0], Y[b, 1], c="g")
   
    ax.scatter(Y[c, 0], Y[c, 1], c="b")
 
    ax.scatter(Y[d, 0], Y[d, 1], c="c")
    ax.scatter(Y[e, 0], Y[e, 1], c="m")
         
    ax.scatter(Y[f, 0], Y[f, 1], c="y")

    ax.scatter(Y[g, 0], Y[g, 1], c="k")
    ax.scatter(Y[h, 0], Y[h, 1], c="g", marker = '^')
    ax.scatter(Y[i, 0], Y[i, 1], c="r", marker = 'x')
 
    ax.scatter(Y[j, 0], Y[j, 1], c="b", marker = 's')
     #ax.scatter(Y[blue, 0], Y[blue, 1], c="b")
   ''' 
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     ax.axis('tight')
#     ax.grid()
# 
#     #plot_results(caracteristicas_wkl, bgmm.predict(caracteristicas_wkl), bgmm.means_, bgmm.covariances_, 1, 'Bayesian Gaussian Mixture with a Dirichlet process prior')
#     #http://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html#sphx-glr-auto-examples-mixture-plot-concentration-prior-py
#     #http://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html#sklearn.mixture.BayesianGaussianMixture
#     #data = np.float32(caracteristicas_wkl)
#     #n_rows, n_cols = 10, 10
#     #som = somoclu.Somoclu(n_cols, n_rows, compactsupport = False)
#     #som.train(data)
#     #som.view_umatrix()
#     
    #%% graficar pupila - clusters

## 
# 
# 
# 
# #%% agglomerative  
#     '''   
#     data = ccs_
#    
#     knn_graph = kneighbors_graph(data, 30, include_self=False)
#     calinski = []
#     labels = []
#     for k in range(2,11):
#         af = AgglomerativeClustering(n_clusters = k,linkage="average", affinity='cityblock').fit(data)  
#         labels_ = af.labels_
#     
#         #s = metrics.silhouette_score(data, labels, metric='euclidean')
#         #silueta.append(s)
#         #print("Silhouette Coefficient: %0.3f" % s )
#         ch = metrics.calinski_harabaz_score(data, labels_)
#         calinski.append(ch)
#         labels.append(labels_)
#         print("Calinsky: " +  str(ch))
#        
#     best_k_ch = max(enumerate(calinski),key=lambda x: x[1])[0] + 2
#     print('Estimated number of clusters: %d' % best_k_ch)
#     
#     '''
# 
# #%% jaccard
# '''
#     for n in range(2):
#         
#         #resample data con bootstrap
#         df_resampled = pd.DataFrame(index=caracteristicas_wkl.index, columns=caracteristicas_wkl.columns)
#         for col in caracteristicas_wkl.columns:
#             df_resampled[col] = cl.bootstrap_resample(caracteristicas_wkl[col])
#         
#         #clusterizar data resampleada
#         af = AgglomerativeClustering(n_clusters =int(n_clusters),linkage="ward", affinity="euclidean", connectivity=knn_graph).fit(df_resampled)  
#         labels_ = af.labels_
#         
#         #para cada cluster, buscar cluster mas similar y calcular jaccard
#     
#         j = metrics.jaccard_similarity_score(labels, labels_) #si tiene distintos numreos? 1 o 3?
# 
# #%% hdbscan
# 
#     clusterer = hdbscan.HDBSCAN(min_cluster_size = 5)
#     cluster_labels = clusterer.fit_predict(data)
#     plt.figure()
#     clusterer.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
#     
#     #s = metrics.calinski_harabaz_score(data, cluster_labels)
#     #print("CH Coefficient: %0.3f"% s )
#     
#     #plt.figure()
#     #plt.plot(cluster_labels, 'r*')
# '''
# =============================================================================
