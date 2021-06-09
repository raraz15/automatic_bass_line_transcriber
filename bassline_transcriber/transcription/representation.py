#!/usr/bin/env python
# coding: utf-8

import numpy as np

from utilities import calcRegionBounds
from .midi_transcription import midi_sequence_to_midi_array
from MIDI_output import create_MIDI_file

# -------------------------------------------- ENCODING -------------------------------------------------------------------------


def encode_midi_sequence(midi_sequence, sustain_code=100, silence_code=0, key=None):
    """Transposes the midi number sequence and inserts sustain code."""

    if key is not None:
        midi_sequence = transpose_to_C(midi_sequence, key, silence_code)

    for idx in range(len(midi_sequence))[::-1][:-1]:
        if midi_sequence[idx] == midi_sequence[idx - 1]:
            midi_sequence[idx] = sustain_code

    return midi_sequence
    

def distance_to_C(key):
    """Returns the number of intervals between the Key and C."""

    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitches.index(key)


def transpose_to_C(midi_sequence, key, silence_code=0):
    """Transposes a given midi sequence to C by calculating root distances. Silences are kept zero."""

    N_intervals = distance_to_C(key) # root notes distance to C in semitones

    midi_array_T = np.array([m-N_intervals if m != silence_code else m for m in midi_sequence ])
    
    return midi_array_T

# --------------------------------------------------- DECODING ------------------------------------------------

def NN_output_to_midi_array(representation, frame_factor, N_qb=8, min_note=28, silence_code=0, sustain_code=100, velocity=120):
    """
    Converts the NN symbolic representation to a MIDI array
    """
    midi_code = representation_to_code(representation, min_note=min_note, silence_code=silence_code, sustain_code=sustain_code)

    midi_code = replace_sustain(midi_code, sustain_code)

    # Silence code will be taken care of
    midi_array = midi_sequence_to_midi_array(midi_code, N_qb, frame_factor, silence_code=silence_code, velocity=velocity)
    
    return midi_array


def representation_to_code(representation, min_note=28, silence_code=0, sustain_code=100):
    """Converts NN output (consecutive integers) to midi code  0, 28,29,..., 51, 100  for example"""

    code = representation.copy()
    
    code[code!=silence_code] += min_note-1

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

    midi_array = NN_output_to_midi_array(representation, frame_factor=M, N_qb=N_qb, min_note=min_note,
                            silence_code=silence_code, sustain_code=sustain_code, velocity=velocity)

    create_MIDI_file(midi_array, BPM, title, output_dir, middle_c=middle_c, tpb=tpb)

def cont_NN_output_to_MIDI_file(representation, title, output_dir, M,
                            BPM=125, N_qb=8, middle_c='C3', tpb=960*16,
                            silence_code=0, velocity=120):

    midi_array = midi_sequence_to_midi_array(representation, M=M, N_qb=N_qb,
                            silence_code=silence_code, velocity=velocity)

    create_MIDI_file(midi_array, BPM, title, output_dir, middle_c=middle_c, tpb=tpb)