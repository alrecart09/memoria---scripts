#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:36:51 2017

@author: antonialarranaga
"""

import os
import funciones as fn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#resultados cluster - matriz que indica cantidad de clusters por persona


path = os.path.dirname(os.path.realpath(__file__))


#cargar resultados
def path_resultados(t):
    path_resultados = path + '/resultados/' + str(t) + '/'
    return path_resultados

#wkl
ccs_wkl_t5 =  pd.read_pickle(path_resultados(5) + 'seleccion_ccs_wklHistograma.pkl')
ccs_wkl_t2 =  pd.read_pickle(path_resultados(2) + 'seleccion_ccs_wklHistograma.pkl')

#valencia
ccs_valencia_sinPupila_t2 = pd.read_pickle(path_resultados(2) + 'seleccion_ccs_valenciaHistogramaSinPupila.pkl')
ccs_valencia_conPupila_t2 = pd.read_pickle(path_resultados(2) + 'seleccion_ccs_valenciaHistogramaConPupila.pkl')

ccs_valencia_sinPupila_t5_1 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_valenciaHistogramaSinPupila_hastaLerko.pkl')
ccs_valencia_sinPupila_t5_2 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_valenciaHistogramaSinPupila_desdeLerko.pkl')

ccs_valencia_conPupila_t5_1 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_valenciaHistogramaConPupila_hastaLerko.pkl')
ccs_valencia_conPupila_t5_2 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_valenciaHistogramaConPupila_desdeLerko.pkl')

#arousal
ccs_arousal_sinPupila_t2 = pd.read_pickle(path_resultados(2) + 'seleccion_ccs_arousalaHistogramaSinPupila.pkl')
ccs_arousal_conPupila_t2 = pd.read_pickle(path_resultados(2) + 'seleccion_ccs_arousalHistogramaConPupila.pkl')

ccs_arousal_sinPupila_t5 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_arousalaHistogramaSinPupila.pkl')
ccs_arousal_conPupila_t5 = pd.read_pickle(path_resultados(5) + 'seleccion_ccs_arousalHistogramaConPupila.pkl')

###################finales####################
#unir valencia
ccs_valencia_conPupila_t5 = pd.concat([ccs_valencia_conPupila_t5_1[:10], ccs_valencia_conPupila_t5_2[:6]] )
ccs_valencia_sinPupila_t5 = pd.concat([ccs_valencia_sinPupila_t5_1[:24], ccs_valencia_sinPupila_t5_2[:13]])#sin lerko y sin cata astorga


#cortar matrices - sujetos con Pupila 16, sin 39
ccs_valencia_sinPupila_t2 = ccs_valencia_sinPupila_t2[:39]
ccs_valencia_conPupila_t2 = ccs_valencia_conPupila_t2[:16]


ccs_arousal_sinPupila_t2 = ccs_arousal_sinPupila_t2[:39]
ccs_arousal_conPupila_t2 = ccs_arousal_conPupila_t2[:16]
ccs_arousal_sinPupila_t5 = ccs_arousal_sinPupila_t5[:39]
ccs_arousal_conPupila_t5 = ccs_arousal_conPupila_t5[:16]

'''
# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots()
sns.heatmap(ccs_arousal_sinPupila_t2, annot=False, fmt="f", ax=ax,  linewidths=.5) #cmap="inferno",
#Colormap husl is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

'''

#sumar y ver lista por orden (mas repetida a menos)

##VALENCIA##
print('\x1b[1;45m Valencia t = 5 \x1b[0m')
total_conPupila = ccs_valencia_conPupila_t5.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
total_sinPupila = ccs_valencia_sinPupila_t5.sum(axis = 0)
total_sinPupila =  total_sinPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas sin pupila\n' + str(total_sinPupila[:6]))
print(' 5 caracteristicas mas seleccionadas con pupila\n' + str(total_conPupila[:6]))

print('\x1b[1;45m Valencia t = 2 \x1b[0m')
total_conPupila = ccs_valencia_conPupila_t2.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
total_sinPupila = ccs_valencia_sinPupila_t2.sum(axis = 0)
total_sinPupila =  total_sinPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas sin pupila\n' + str(total_sinPupila[:6]))
print(' 5 caracteristicas mas seleccionadas con pupila\n' + str(total_conPupila[:6]))


##AROUSAL##
print('\x1b[1;45m Arousal t = 5 \x1b[0m')
total_conPupila = ccs_arousal_conPupila_t5.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
total_sinPupila = ccs_arousal_sinPupila_t5.sum(axis = 0)
total_sinPupila =  total_sinPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas sin pupila\n' + str(total_sinPupila[:6]))
print(' 5 caracteristicas mas seleccionadas con pupila\n' + str(total_conPupila[:6]))

print('\x1b[1;45m Arousal t = 2 \x1b[0m')
total_conPupila = ccs_arousal_conPupila_t2.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
total_sinPupila = ccs_arousal_sinPupila_t2.sum(axis = 0)
total_sinPupila =  total_sinPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas sin pupila\n' + str(total_sinPupila[:6]))
print(' 5 caracteristicas mas seleccionadas con pupila\n' + str(total_conPupila[:6]))

####WKL####

print('\x1b[1;45m wkl t = 5 \x1b[0m')
total_conPupila = ccs_wkl_t5.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas\n' + str(total_sinPupila[:6]))

print('\x1b[1;45m wkl t = 2 \x1b[0m')
total_conPupila = ccs_wkl_t2.sum(axis = 0)
total_conPupila = total_conPupila.sort_values(ascending = False)
print(' 5 caracteristicas mas seleccionadas\n' + str(total_sinPupila[:6]))


