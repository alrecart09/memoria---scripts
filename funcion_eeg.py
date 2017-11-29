#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:24:09 2017

@author: antonialarranaga
"""

from __future__ import absolute_import, division, print_function
from six.moves import range
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from biosppy import tools as st
from biosppy import utils
from biosppy.plotting import _plot_multichannel
from biosppy.signals.eeg import _power_features

def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    '''
    #Make copy so original not edited
    vals=vals_orig.copy()    
    #Hampel Filter
    L= 1.4826
    rolling_median=vals.rolling(k).median()
    difference=np.abs(rolling_median-vals)
    median_abs_deviation=difference.rolling(k).median()
    threshold= t0 *L * median_abs_deviation
    outlier_idx=difference>threshold
    vals[outlier_idx]=rolling_median[outlier_idx]
    return(vals)

#funcion basada en eeg de biosppy, pero entrega menos cosas y cambia el rango de frecuencias
def eeg2(signal, canales, sampling_rate, labels = None, show=True):
    """Pasa por pasabanda de 4-40 Hz y entrega las bandas de frecuencia para cada canal
    
    Parameters
    ----------
    signal : array
        Raw EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    labels : list, optional
        Channel labels.
    show : bool, optional
        If True, show a summary plot.
    
    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered BVP signal.    
    """
    # check inputs
    if signal is None:
        raise TypeError('No hay señal de entrada')
        
    ch_nombre = list(signal.columns.values)
    # ensure numpy
    signal = np.array(signal)
    
    sampling_rate = float(sampling_rate)
    nch = canales
    
    if labels is None:
        labels = ['Ch. %d' % i for i in range(nch)]
    else:
        if len(labels) != nch:
            raise ValueError(
                "Number of channels mismatch between signal matrix and labels.")
    
    # high pass filter
    b, a = st.get_filter(ftype='butter',
                         band='highpass',
                         order=8,
                         frequency=0.1,
                         sampling_rate=sampling_rate)
    
    aux, _ = st._filter_signal(b, a, signal=signal, check_phase=True, axis=0)
    
    # low pass filter
    b, a = st.get_filter(ftype='butter',
                         band='lowpass',
                         order=16,
                         frequency=40,
                         sampling_rate=sampling_rate)
    
    filtered, _ = st._filter_signal(b, a, signal=aux, check_phase=True, axis=0)
    hampel_eeg = pd.DataFrame(data = filtered, columns = ch_nombre)
    for j in ch_nombre:
        hampel_eeg[j] = hampel(hampel_eeg[j], k=7, t0= 3)
    # band power features
    hampel_eeg = np.array(hampel_eeg)

    '''
    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    # output
    args = (ts, filtered, hampel_eeg)
    names = ('ts', 'filtered', 'hampel')
    
    return utils.ReturnTuple(args, names)
    
    '''
    return filtered, hampel_eeg
    
    
    #modificacion de la dada por biosppy
def get_power_features2(signal=None, channel = 14,
                       sampling_rate=1000.,
                       size=0.25,
                       overlap=0.5):
    """Extract band power features from EEG signals.
    
    Computes the average signal power, with overlapping windows, in typical
    EEG frequency bands:
    * Theta: from 4 to 8 Hz,
    *Alpha: from 8 to 12Hz,
    * Higher Alpha: from 10 to 13 Hz,
    * Beta: from 13 to 25 Hz,
    * Gamma: from 25 to 40 Hz.
    
    Parameters
    ----------
    signal  array
        Filtered EEG signal matrix; each column is one EEG channel.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    size : float, optional
        Window size (seconds).
    overlap : float, optional
        Window overlap (0 to 1).
    
    Returns
    -------
    ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one EEG
        channel.
    alpha: array
        Average power in the 8 to 12 Hz frequency band; each column is one EEG
        channel.
    beta : array
        Average power in the 12 to 25 Hz frequency band; each column is one EEG
        channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one EEG
        channel.
    
    """
    
    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")
    
    # ensure numpy
    signal = np.array(signal)
    nch = channel
    
    sampling_rate = float(sampling_rate)
    
    # convert sizes to samples
    size = int(size * sampling_rate)
    step = size - int(overlap * size)
    
    # padding
    min_pad = 1024
    pad = None
    if size < min_pad:
        pad = min_pad - size
    
    # frequency bands
    bands = [[4, 8], [8, 12], [13, 25], [25, 40]]
    nb = len(bands)
    
    # windower
    fcn_kwargs = {'sampling_rate': sampling_rate, 'bands': bands, 'pad': pad}
    index, values = st.windower(signal=signal,
                                size=size,
                                step=step,
                                kernel='hann',
                                fcn=_power_features,
                                fcn_kwargs=fcn_kwargs)
    
    # median filter
    md_size = int(0.625 * sampling_rate / float(step))
    if md_size % 2 == 0:
        # must be odd
        md_size += 1
    
    for i in range(nb):
        for j in range(nch):
            values[:, i, j], _ = st.smoother(signal=values[:, i, j],
                                             kernel='median',
                                             size=md_size)
    
    # extract individual bands
    theta = values[:, 0, :]
    alpha = values[:, 1, :]
    beta = values[:, 2, :]
    gamma = values[:, 3, :]
    
    # convert indices to seconds
    ts = index.astype('float') / sampling_rate
    
    # output
    args = (ts, theta, alpha, beta, gamma)
    names = ('ts', 'theta', 'alpha', 'beta', 'gamma')
    
    return utils.ReturnTuple(args, names)

def plot_eeg2(ts=None,
             raw=None,
             filtered=None,
             labels=None,
             features_ts=None,
             theta=None,
             alpha=None,
             beta=None,
             gamma=None,
             path=None,
             show=False):
    """Create a summary plot from the output of signals.eeg.eeg.

    Parameters
    ----------
    ts : array
        Signal time axis reference (seconds).
    raw : array
        Raw EEG signal.
    filtered : array
        Filtered EEG signal.
    labels : list
        Channel labels.
    features_ts : array
        Features time axis reference (seconds).
    theta : array
        Average power in the 4 to 8 Hz frequency band; each column is one
        EEG channel.
    alpha: array
        Average power in the 8-12 Hz frequency band; each column is one
        EEG channel.
    beta : array
        Average power in the 13 to 25 Hz frequency band; each column is one
        EEG channel.
    gamma : array
        Average power in the 25 to 40 Hz frequency band; each column is one
        EEG channel.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show the plot immediately.

    """

    nrows = 10
    alpha_ = 2.

    figs = []

    # raw
    fig = _plot_multichannel(ts=ts,
                             signal=raw,
                             labels=labels,
                             nrows=nrows,
                             alpha=alpha_,
                             title='EEG Summary - Raw',
                             xlabel='Time (s)',
                             ylabel='Amplitude')
    figs.append(('_Raw', fig))

    # filtered
    fig = _plot_multichannel(ts=ts,
                             signal=filtered,
                             labels=labels,
                             nrows=nrows,
                             alpha=alpha_,
                             title='EEG Summary - Filtered',
                             xlabel='Time (s)',
                             ylabel='Amplitude')
    figs.append(('_Filtered', fig))

    # band-power
    names = ('Theta Band', 'Alpha Band',
             'Beta Band', 'Gamma Band')
    args = (theta, alpha, beta, gamma)
    for n, a in zip(names, args):
        fig = _plot_multichannel(ts=features_ts,
                                 signal=a,
                                 labels=labels,
                                 nrows=nrows,
                                 alpha=alpha_,
                                 title='EEG Summary - %s' % n,
                                 xlabel='Time (s)',
                                 ylabel='Power')
        figs.append(('_' + n.replace(' ', '_'), fig))

    # save to file
    if path is not None:
        path = utils.normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            ext = '.png'

        for n, fig in figs:
            path = root + n + ext
            fig.savefig(path, dpi=200, bbox_inches='tight')

    # show
    if show:
        plt.show()
    else:
        # close
        for _, fig in figs:
            plt.close(fig)