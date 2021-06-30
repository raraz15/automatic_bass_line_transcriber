#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np

from MIDI_output import create_MIDI_file
from bassline_transcriber.transcription import midi_sequence_to_midi_array, downsample_midi_sequence

# -------------------------------------------- ENCODING ----------------------------------------------------------------------
# pitch_track = Hz > midi numbers > |insert_silence_code| > midi_sequence > |transpose| > |put_sustain|  > code > |make consecutive| > representation > |NN| 

def encode_midi_sequence(midi_sequence, key, M, N_qb=8, sustain_code=100, silence_code=0, MAX_NOTE=51, MIN_NOTE=28):
    """Encodes a midi sequence to be used by the Neural Network.
        
        Parameters:
        -----------
            midi_sequence (ndarray): the midi sequence of the pitch track with silences indicated by the silence code.
            key (str): the key e.g. A#
            M (int): decimation rate to be applied to the midi sequence
            N_qb (int, default=8): 
            sustain_code (int default=100): 
        
        Returns:
        --------
            None if the midi sequence contains unwanted midi notes."""

    midi_sequence = downsample_midi_sequence(midi_sequence, M, N_qb=N_qb)

    midi_sequence = transpose_to_C(midi_sequence, key, silence_code)

    if pre_sustain_filter(midi_sequence, silence_code=silence_code, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE):

        midi_sequence = put_sustain(midi_sequence, sustain_code)

        midi_sequence = make_consecutive_symbols(midi_sequence, sustain_code=sustain_code, silence_code=silence_code, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE)

        return midi_sequence

    else:

        return None

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


    #if X[np.where((X>MAX_NOTE) & (X!=sustain_code))].size >0:
    #    return None
    #if X[np.where((X<MIN_NOTE)&(X!=silence_code))].size >0:
    #    return None
    #else:


def pre_sustain_filter(X, MAX_NOTE=51, MIN_NOTE=28, silence_code=0):
    """"Filter before putting the sustain."""

    flag = True

    if X[X>MAX_NOTE].size > 0:
        flag = False
        return flag

    y = X[X<MIN_NOTE]
    if y[y!=silence_code].size > 0:
        flag = False
        return flag

    return flag


def make_consecutive_symbols(X, sustain_code=100, silence_code=0, MAX_NOTE=51, MIN_NOTE=28):
    """Make the symbols consecutive integers."""

    X[X==sustain_code] = MAX_NOTE+1
    X[X!=silence_code] -= MIN_NOTE-1
    X[X==silence_code] = 0 

    return X

# --------------------------------------------------- DECODING ------------------------------------------------

def NN_output_to_midi_array(representation, frame_factor, N_qb=8, min_note=28, silence_code=0, sustain_code=100, velocity=120):
    """Converts the NN symbolic representation to a MIDI array"""

    midi_code = representation_to_code(representation, min_note=min_note, silence_code=silence_code, sustain_code=sustain_code)

    if sustain_code is not None:
        midi_code = replace_sustain(midi_code, sustain_code)

    # Silence code will be taken care of
    midi_array = midi_sequence_to_midi_array(midi_code, N_qb, frame_factor, silence_code=silence_code, velocity=velocity)
    
    return midi_array


def representation_to_code(representation, min_note=28, silence_code=0, sustain_code=100):
    """Converts NN output (consecutive integers) to midi code  0, 28,29,..., 51, 100  for example"""

    code = representation.copy()
    code[code!=silence_code] += min_note-1
    if sustain_code is not None:
        max_code = code.max() 
        code[code==max_code] = sustain_code # max will be the sustain
    return code


def replace_sustain(codes, sustain_code=100):

    arr = codes.copy()
    if len(arr.shape) == 2:
        for r in arr:
            for idx, el in enumerate(r[1:]):
                if el == sustain_code:
                    r[idx + 1] = r[idx]
    else:
        for idx, el in enumerate(arr[1:]):
            if el == sustain_code:
                arr[idx + 1] =arr[idx]        

    return arr

# --------------------------------------------------- MIDI -------------------------------------------------

def NN_output_to_MIDI_file(representation, title, output_dir, M, 
                            BPM=125, N_qb=8, middle_c='C3', tpb=960*16,
                            min_note=28, silence_code=0, sustain_code=100, velocity=120):

    os.makedirs(output_dir, exist_ok=True)

    _representation = representation.copy()

    midi_array = NN_output_to_midi_array(_representation, frame_factor=M, N_qb=N_qb, min_note=min_note,
                            silence_code=silence_code, sustain_code=sustain_code, velocity=velocity)

    create_MIDI_file(midi_array, BPM, title, output_dir, middle_c=middle_c, tpb=tpb)