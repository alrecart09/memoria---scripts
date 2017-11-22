#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:40:15 2017

@author: antonialarranaga
"""
import scipy.io
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import funciones as fn
import pyfftw


path = os.path.dirname(os.path.realpath(__file__)) 
sujeto = 'tomas-lagos'
nombre = '/sujetos/' + sujeto + '/'

df = pd.read_pickle(path + nombre + sujeto + '_syncGSRPPG.pkl')

data = df['GSR_PPG_PPG_A13_CAL']
t = df['GSR_PPG_TimestampSync_Unix_CAL']
'''
data = df['GSR_PPG_PPG_A13_CAL']
t = df['GSR_PPG_TimestampSync_Unix_CAL']

plt.subplot(2, 1, 2)
plt.plot(t, data, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
'''

#sacar fft de ppg 
Fs = 120

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
#t = np.arange(0,1,Ts) # time vector

#ff = 5;   # frequency of the signal
#data = np.sin(2*np.pi*ff*t)

a = pyfftw.empty_aligned(data.size, dtype='complex128')
b = pyfftw.empty_aligned(data.size, dtype='complex128')
fft_obj = pyfftw.FFTW(a, b)
a[:] = data

fft_a = fft_obj()

c = pyfftw.empty_aligned(data.size, dtype = 'complex128')


n = len(a)

k = np.arange(n)
T = n/Fs
freq = k/T # two sides frequency range
#freq = freq[range(int(n/2))] # one side frequency range

#b = pyfftw.interfaces.numpy_fft.fft(a) # fft computing and normalization

#b = b[range(int(n/2))]

'''
fig, ax = plt.subplots(2, 1)
ax[0].plot(t,data)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(b),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
'''

fc1 = 0.5
fc2 = 10

espectro = abs(fft_a)

fourier = np.column_stack([freq, fft_a])
#magnitud vs freq

for index, elemento in enumerate(fourier):
    if 0<abs(elemento[0])<fc1:
#        print(f, e)
        fourier[index][1]= 0
#        print(f, e)
    if abs(elemento[0])>fc2:
        fourier[index][1] = 0

b[:] = fourier[0:data.size,1]

ifft_obj = pyfftw.FFTW(b, c, direction = 'FFTW_BACKWARD')

#plt.plot(freq, abs(fourier[0:234576,1]))

data_limpia2 = ifft_obj()
data_limpia = pyfftw.interfaces.numpy_fft.ifft(b)



#plt.set_xlabel('Time')
#plt.set_ylabel('Amplitude')
plt.plot(t,data)
#plt.plot(t,abs(data_limpia2), 'r') # plotting the spectrum
plt.plot(t, abs(data_limpia2), 'g')
plt.show()


'''
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

Fs = 150.0;  # sampling rate
Ts = 1.0/Fs; # sampling interval
t = np.arange(0,1,Ts) # time vector

ff = 5;   # frequency of the signal
y = np.sin(2*np.pi*ff*t)

n = len(y) # length of the signal
k = np.arange(n)
T = n/Fs
frq = k/T # two sides frequency range
frq = frq[range(n/2)] # one side frequency range

Y = np.fft.fft(y)/n # fft computing and normalization
Y = Y[range(n/2)]

fig, ax = plt.subplots(2, 1)
ax[0].plot(t,y)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Amplitude')
ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')

plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')
'''