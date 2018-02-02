#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:02:21 2018

@author: antonialarranaga
"""

#deep learning

from sknn.mlp import Layer, Classifier
from sknn.platform import cpu32, threading
import funciones as fn
import pandas as pd
import collections 
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import fn_clasificar as clsf
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score


def funcion_clasificar(x, y, clasificador, nombre, nClases, num_trials=30):
    accuracy_i= []
    precision_i= []
    recall_i= []
    f1_i = []
    for i in range(num_trials):
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, shuffle = True, stratify = y)

        pipe_ = Pipeline([('estandarizar', StandardScaler()), (nombre, clasificador)])
        pipe_.fit(xTrain, yTrain)
        yPredichas = pipe_.predict(xTest)
        accuracy = accuracy_score(yTest, yPredichas)
        #print(classification_report(yTest, yPredichas))
        precision, recall, fscore, _ = precision_recall_fscore_support(yTest, yPredichas, average = 'macro')
        
        accuracy_i.append(accuracy)
        precision_i.append(precision)
        recall_i.append(recall)
        f1_i.append(fscore)
    
    ac = [np.array(accuracy_i).mean(), np.array(accuracy_i).std()]
    pre = [np.array(precision_i).mean(), np.array(precision_i).std()]
    rec =[np.array(recall_i).mean(), np.array(recall_i).std()] 
    f =  [np.array(f1_i).mean(), np.array(f1_i).std()]
    return ac, pre, rec, f

path = '/Volumes/ADATA CH11'

#path = os.path.dirname(os.path.realpath(__file__))


t = 2
#participantes = fn.listaParticipantes()[0]

num_repeticiones = 5
warnings.filterwarnings('ignore') 
#participantes = []

participantes = ['todos']
path_resultados = fn.makedir2(path, 'resultados/' + str(t) )
clasificaciones = ['neu_300', 'neu_500', 'neu_1000', 'neu_3000', 'neu_5000']

c_acc = [s + '_acc' for s in clasificaciones]
c_acc_std = [s + '_accStd' for s in clasificaciones]
c_prec = [s + '_prec' for s in clasificaciones]
c_prec_std = [s + '_precStd' for s in clasificaciones]
c_rec = [s + '_rec' for s in clasificaciones]
c_rec_std = [s + '_recStd' for s in clasificaciones]
c_f1 = [s + '_f1' for s in clasificaciones]
c_f1_std = [s + '_f1Std' for s in clasificaciones]

clmn = []
for i in range(len(clasificaciones)):
    clmn.append(c_acc[i])
    clmn.append(c_acc_std[i])
    clmn.append(c_prec[i])
    clmn.append(c_prec_std[i])
    clmn.append(c_rec[i])
    clmn.append(c_rec_std[i])
    clmn.append(c_f1[i])
    clmn.append(c_f1_std[i])
    
matrix = np.empty(shape = (len(participantes), len(clmn)))    #len(clmn)
i=0    

for sujeto in participantes:
    print('\x1b[1;45m' + str(sujeto) +'\x1b[0m')

    resultados = []
    
        
    path_ccs = path +'/señales_baseline/ccs_todosWKL.pkl'
    
    ccs_wkl = pd.read_pickle(path_ccs)
    

    path_etiquetas = path + '/clusters_todosWKL/' + str(t) 
    
    caracteristicas = ccs_wkl #ccs_a, ccs_v
    etiquetas = pd.read_pickle(path_etiquetas + '/' + sujeto + 'clusters.pkl') #_etiquetas-arousalGSR.pkl

    
    caracteristicas.reset_index(drop = True, inplace = True) #tengan mismos indices - partan de 0 hasta len
    etiquetas.reset_index(drop = True, inplace = True)
    
    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    print('numero original de clases: ' + str(len(repeticion)))
    
    for clase, cantidad in repeticion:
       # print(str(clase) + ' ' + str(cantidad))
        if cantidad < 6:
            #print('clase ' + str(clase) + ' con <= de 5 elementos')
            
            indices = np.where(etiquetas == clase)[0]
            #print(indices)
            caracteristicas = caracteristicas.drop(list(indices))
            etiquetas = etiquetas.drop(list(indices))
            
            caracteristicas.reset_index(drop = True, inplace = True)
            etiquetas.reset_index(drop = True, inplace = True)
            #print('borre clase ' + str(clase))
    
    cuenta = collections.Counter(etiquetas.values.ravel())
    repeticion = cuenta.most_common()
    
    if len(repeticion) == 1: #revisar que queden más de una clase
        print('no se puede hacer clasificacion, quedó una clase :(')
        continue
    
    print('numero final de clases: ' + str(len(repeticion)))      
  
    caracteristicas = np.array(caracteristicas)
    etiquetas = etiquetas.values.ravel()
    
    
    
    print('ANN')
    neuronas_o = [300, 500, 1000, 3000, 5000]
    for n_ocultas in neuronas_o:
        print(' neuronas ocultas:' + str(n_ocultas))
        ann = Classifier(layers=[  #For hidden layers, you can use the following layer types: Rectifier, ExpLin, Sigmoid, Tanh, or Convolution.
                Layer("Rectifier", units=n_ocultas),
                Layer("Rectifier", units=n_ocultas),
                Layer("Softmax", units= len(repeticion))], #For output layers, you can use the following layer types: Linear or Softmax.
            n_iter=250,
            n_stable=10,
            batch_size=25,
            learning_rate=0.002,
            regularize = 'dropout',
            #normalize = 'batch',
            learning_rule="momentum", #sgd, momentum, nesterov, adadelta, adagrad or rmsprop
            valid_size=0.1, #validacion 0.1 de training
            dropout_rate = 0.5)
        accuracy, precision, recall, f1 = funcion_clasificar(caracteristicas, etiquetas, ann, 'clasificador', len(repeticion), num_trials=num_repeticiones)
        clsf.guardar_resultados(accuracy, precision, recall, f1, resultados)
        print('acc = ', accuracy, ' - f1 = ', f1)
    matrix[i, :] = resultados
    
i +=1
    
df_resultados = pd.DataFrame(matrix, columns = clmn)

