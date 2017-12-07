#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:10:19 2017

@author: antonialarranaga
"""
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.feature_selection import RFECV

#seleccionar características con RFE-RF
def rfecvRF(x, etiquetas):
    #data = DF matriz de obsxccs, etiquetas = labels
      
    class RandomForestClassifierWithCoef(RandomForestClassifier):
        @property
        def feature_importances_(self):
          # print('hola soy RFCC')
          # print(super().feature_importances_)
            return stats.zscore(super().feature_importances_)
    
    y=etiquetas.values.ravel() #revisar cantidad de etiquetas?
    rf = RandomForestClassifierWithCoef(n_estimators = 100, oob_score = True, n_jobs=-1)
    rfecv = RFECV(estimator=rf, step=1, cv=2, verbose=0)
    selector=rfecv.fit(x, y)
    
    return selector #que deveulve?
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html lo que devuelve el selector


'''
def lista_rfeRF(data, etiquetas, caracteristicas):
    #data = num_samples x num_ccs
    #etiquetas = num_samples
    #caracteristicas = array con nombre de caracteristicas en orden
    num_ccs = data.size[1] #revisar si es la dimension correcta
    X = data
    y = etiquetas
    I = []
    while num_ccs:
        clf = RandomForestClassifier(oob_score = True, n_jobs = -1) #revisar max_feature = None
        clf.fit(X,y)
        importance = pd.DataFrame({'features': caracteristicas, 'imp': clf.feature_importances_})
        #clf_feature_importances_ toma en cuenta todas las caracteristicas?? revisar!!
        importance['imp'] = stats.zscore(importance['imp']) #paso a ct - revisar paper zscore **
        importance = importance.sort_values(by='imp', ascending=False) #ver si indices se re-arreglan
        I.append(importance.index[num_ccs-1]) #no se si se puede hacer asi
        num_col = np.nonzero(caracteristicas == importance['features'][num_ccs-1]) #está en row = ultima, col = segunda
        importance = importance.drop(num_ccs-1) # DF = DF[:-1]
        X = np.delete(X, num_col, 1) #sacarle la columna correspondiente a la caracteristica que elimine
        num_ccs -= 1
    return I.reverse() #lista con caracteristicas de mayor a menor (primera la mas imp)
        
def rfeRF(data, etiquetas, caracteristicas, num_ccs):
    lista = lista_rfeRF(data, etiquetas, caracteristicas)
    return lista[:num_ccs] #subindex it with [:5] indicating that you want (up to) the first 5 elements.
'''