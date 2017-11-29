#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:47:37 2017

@author: antonialarranaga
"""
from biosppy import tools as st
from biosppy import utils
import scipy.signal as ss

def filter_signal2(signal=None,
                  ftype='FIR',
                  band='lowpass',
                  order=None,
                  frequency=None,
                  sampling_rate=1000., **kwargs):
    """Filter a signal according to the given parameters.

    Parameters
    ----------
    signal : array
        Signal to filter.
    ftype : str
        Filter type:
            * Finite Impulse Response filter ('FIR');
            * Butterworth filter ('butter');
            * Chebyshev filters ('cheby1', 'cheby2');
            * Elliptic filter ('ellip');
            * Bessel filter ('bessel').
    band : str
        Band type:
            * Low-pass filter ('lowpass');
            * High-pass filter ('highpass');
            * Band-pass filter ('bandpass');
            * Band-stop filter ('bandstop').
    order : int
        Order of the filter.
    frequency : int, float, list, array
        Cutoff frequencies; format depends on type of band:
            * 'lowpass' or 'bandpass': single frequency;
            * 'bandpass' or 'bandstop': pair of frequencies.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying
        scipy.signal function.

    Returns
    -------
    signal : array
        Filtered signal.
    sampling_rate : float
        Sampling frequency (Hz).
    params : dict
        Filter parameters.

    Notes
    -----
    * Uses a forward-backward filter implementation. Therefore, the combined
      filter has linear phase.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify a signal to filter.")

    # get filter
    b, a = st.get_filter(ftype=ftype,
                      order=order,
                      frequency=frequency,
                      sampling_rate=sampling_rate,
                      band=band, **kwargs)

    # filter
    filtered, _ = _filter_signal2(b, a, signal, check_phase=True)

    # output
    params = {
        'ftype': ftype,
        'order': order,
        'frequency': frequency,
        'band': band,
    }
    params.update(kwargs)

    args = (filtered, sampling_rate, params)
    names = ('signal', 'sampling_rate', 'params')

    return utils.ReturnTuple(args, names)

def _filter_signal2(b, a, signal, zi=None, check_phase=True, **kwargs):
    """Filter a signal with given coefficients.

    Parameters
    ----------
    b : array
        Numerator coefficients.
    a : array
        Denominator coefficients.
    signal : array
        Signal to filter.
    zi : array, optional
        Initial filter state.
    check_phase : bool, optional
        If True, use the forward-backward technique.
    ``**kwargs`` : dict, optional
        Additional keyword arguments are passed to the underlying filtering
        function.

    Returns
    -------
    filtered : array
        Filtered signal.
    zf : array
        Final filter state.

    Notes
    -----
    * If check_phase is True, zi cannot be set.

    """

    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True.")

    padle_pred = max(len(a), len(b))*3
    
    if padle_pred < len(signal):
        if zi is None:
            zf = None
            if check_phase:
                filtered = ss.filtfilt(b, a, signal, **kwargs)
            else:
                filtered = ss.lfilter(b, a, signal, **kwargs)
        else:
            filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)
    
    else: #padle es mayor tamaño señal
        padlen = len(signal)-10
        if zi is None:
            zf = None
            if check_phase:
                filtered = ss.filtfilt(b, a, signal, padlen = padlen, **kwargs)
            else:
                filtered = ss.lfilter(b, a, signal, **kwargs)
        else:
            filtered, zf = ss.lfilter(b, a, signal, zi=zi, **kwargs)

    return filtered, zf

