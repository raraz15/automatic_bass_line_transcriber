#!/usr/bin/env python
# coding: utf-8

import numpy as np

from ..bass_line_transcriber.transcription import  downsample_midi_sequence


# pitch_track = Hz > midi numbers > |insert_silence_code| > midi_sequence > |downsample| > |transpose| >
# |put_sustain|  > code > |make consecutive| > representation > |NN| 


def encode_midi_sequence(midi_sequence, key, M, N_qb=8, sustain_code=100, silence_code=0, MIN_NOTE=28, MAX_NOTE=51):
    """Encodes a midi sequence to be used by the Neural Network.
        
        Parameters:
        -----------
            midi_sequence (ndarray): midi sequence corresponding to a pitch track, silence indicated by the silence code.
            key (str): the key e.g. A# (no min or maj indicator)
            M (int): decimation rate to be applied to the midi sequence
            N_qb (int, default=8): 
            sustain_code (int default=100): 

        Returns:
        --------
            representation None if the midi sequence contains unwanted midi notes or ndarray."""

    representation = midi_sequence.copy()
    representation = downsample_midi_sequence(representation, M, N_qb=N_qb)
    representation = transpose_to_C(representation, key, silence_code)
    if code_filter(representation, MIN_NOTE=MIN_NOTE, MAX_NOTE=MAX_NOTE, silence_code=silence_code):
        if sustain_code is not None:
            representation = put_sustain(representation, sustain_code)
        representation = make_consecutive_symbols(representation, 
                                                sustain_code=sustain_code,
                                                silence_code=silence_code,
                                                MAX_NOTE=MAX_NOTE,
                                                MIN_NOTE=MIN_NOTE)
    else:
        representation = None  
    return representation


def put_sustain(sequence, sustain_code=100):
    for idx in range(len(sequence))[::-1][:-1]:
        if sequence[idx] == sequence[idx - 1]:
            sequence[idx] = sustain_code
    return sequence


def transpose_to_C(midi_sequence, key, silence_code=0):
    """Transposes a given midi sequence to C by calculating root distances. Silences are kept zero."""

    def distance_to_C(key):
        """Returns the number of intervals between the Key and C."""

        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return pitches.index(key)

    N_intervals = distance_to_C(key) # root notes distance to C in semitones

    midi_array_T = np.array([m-N_intervals if m != silence_code else m for m in midi_sequence])
    
    return midi_array_T


def code_filter(X, MIN_NOTE=28, MAX_NOTE=51, silence_code=0):
    """"Filter out representations (before putting the sustain)."""
    flag = True
    if X[X>MAX_NOTE].size > 0:
        flag = False
        return flag

    y = X[X<MIN_NOTE]
    if y[y!=silence_code].size > 0:
        flag = False
    return flag


def make_consecutive_symbols(X, sustain_code=100, silence_code=0, MAX_NOTE=51, MIN_NOTE=28):
    """Make the symbols consecutive integers."""
    if sustain_code is not None:
        X[X==sustain_code] = MAX_NOTE+1     
    X[X!=silence_code] -= MIN_NOTE-1
    X[X==silence_code] = 0
    return X