#!/usr/bin/env python
# coding: utf-8

import numpy as np

from librosa import stft, amplitude_to_db
from librosa.util import normalize

from scipy.signal import firwin, convolve


def extract_dB_spectrogram(audio, n_fft, win_length, hop_length, center=True):

    assert win_length < n_fft, 'Window length must be greater than N_fft'
    
    amplitude_spectrogram = np.abs(stft(audio, 
                                        n_fft=n_fft, 
                                        win_length=win_length, 
                                        hop_length=hop_length,
                                        center=center))
                                           
    return amplitude_to_db(amplitude_spectrogram, np.max(amplitude_spectrogram))


def lp_and_normalize(track, fc, fs, M=5001, window_type='blackman'):
    """
    Low Pass filters the track with same length convolution and normalizes the output.
    Causal, Generalized Phase Type I filter.

        Parameters:
        -----------
            track(ndarray): audio track
            fc (float): Cut-off frequency in Hz
            fs (float): Sampling frequency in Hz
            N (int, default=5001): Filter tap
            window_type (str, default='blackmann'): window type
        
        Returns:
        --------
            track_cut (ndarray): processed track
    """

    # Type I filter
    lp = firwin(M,
                cutoff=fc,
                window=window_type,
                fs=fs) 

    track_cut = convolve(track, lp, mode='same') # same length convolution

    track_cut = normalize(track_cut) # normalize

    return track_cut


