#!/usr/bin/env python
# coding: utf-8

import numpy as np

from librosa import pyin
from crepe import predict as crepe_predict

from ...utilities import create_frequency_bins

F_MIN = 32.7
F_MAX = 123.47
T_MAX = 1/F_MIN # Longesr Period
FS = 44100


# TODO: confidence filter with numpy features
def pYIN_F0(audio, beat_duration, hop_factor=32, N_bars=4, threshold='none'):
    """

        beat_duration (float): Duration of a beat in seconds
        hop_factor (int): Number of F0 samples that make up a beat.
    """

    frame_length = int(T_MAX*FS)
    hop_length = int((beat_duration/hop_factor)*FS)

    # Number of F0 samples a quarter beat includes
    N_qb = int(hop_factor/4)    
    
    #frame_length = int((beat_duration/frame_factor)*FS)
    #hop_length = int(frame_length/4) # /4 by default, enforce precise length

    F0, _, confidence = pyin(audio,
                            sr=FS,
                            frame_length=frame_length,
                            hop_length=hop_length,
                            fmin=F_MIN-1,
                            fmax=F_MAX+1,
                            fill_na=0.0)

    if threshold == 'none':
        threshold = 0
    elif threshold == 'mean':
        threshold = np.mean(confidence)
    elif threshold == 'mean_reduced':
        threshold = np.mean(confidence) - np.std(confidence)/2
    else:
        assert threshold < 1.0 and threshold > 0, 'Threshold must be in (0, 1)'

    F0 = np.round(F0, 2) # round to 2 decimals

    F0 = ensure_sequence_length(F0, N_qb=N_qb, N_bars=N_bars)

    F0_filtered = confidence_filter(F0, confidence, threshold)

    time_axis = np.arange(len(F0_filtered)) * (hop_length/FS)

    return (time_axis, F0), (time_axis, F0_filtered)


def confidence_filter(F0, confidence, threshold):
    """
    Silences the time instants where the model confidence is below the given threshold.
    """

    return np.array([f if confidence[idx] >= threshold else 0.0 for idx, f in enumerate(F0)])


def ensure_sequence_length(sequence, N_qb=8, N_bars=4):
    """Makes sure that the sequence has proper length."""

    N_required = (N_bars*4)*4*N_qb  # required input signal length

    if len(sequence) < N_required:  # pad if needed
        sequence = np.append(sequence, sequence[-(N_required-len(sequence)):])
    else: # or truncate
        sequence = sequence[:N_required]

    return sequence


def argmax_F0(spectrogram, FS, hop_length):
    
    time_axis = np.arange(spectrogram.shape[1]) * (hop_length/FS)
    
    frequency_bins, _ =  create_frequency_bins(FS, spectrogram.shape[0])
    max_freq_bins = np.argmax(spectrogram, axis=0) # get the frequency bins with highest energy
        
    F0 = np.array([frequency_bins[idx] for idx in max_freq_bins])

    return (time_axis, F0)


def crepe_F0(audio, FS, viterbi=True, threshold='none'):
      
    time_axis, F0, confidence, _ = crepe_predict(audio, FS, viterbi=True)
    
    mean, std = np.mean(confidence), np.std(confidence)
    print('Mean of the confidence levels: {:.3f}\nStandard Deviation: {:.3f}'.format(mean, std))
    
    if threshold == 'none':
        threshold = 0
    elif threshold == 'mean':
        threshold = mean
    elif threshold == 'mean_reduced':
        threshold = mean - (std/2)
    else:
        assert threshold < 1.0 and threshold > 0, 'Threshold must be inside (0, 1)'

    F0_filtered = confidence_filter(F0, confidence, threshold)

    return (time_axis, F0), (time_axis, F0_filtered)