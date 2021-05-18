#!/usr/bin/env python
# coding: utf-8

import numpy as np

from utilities import calcRegionBounds
from .midi_transcription import midi_number_to_midi_array


def encode_midi_array(midi_array, N_qb, M, N_bars):

    filled_midi_array = fill_midi_array(midi_array, N_bars)

    symbolic_representation = convert_to_vector(filled_midi_array, N_qb, M)

    return symbolic_representation


def fill_midi_array(midi_array, N_bars):
    """
    Fills the midi_array silence indices and corrects the endpoints for encoding stage.
        
        Parameters:
        -----------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration]]
            N_bars (int): Number of bars that the midi_array corresponds to

        Returns:
        --------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration]]

    """
    
    indices, rows = [], [] # indices for silence insertion and the notes to insert

    for idx, midi in enumerate(midi_array[:-1]):

        end = midi[0] + midi[3] # start_idx + duration

        if end != midi_array[idx+1, 0]: # if there is a gap between the next note
            indices.append(idx+1) 
            rows.append([end, 0, 120, midi_array[idx+1, 0]-end]) # fill the gap with 0

    if len(indices) > 0: # if there are gaps, fill
        midi_array = np.insert(midi_array, indices, rows, axis=0)


    # Ensure beginning and end indices
    if midi_array[0,0] != 0.0:
        midi_array=np.insert(midi_array, 0, [0.0, 0, midi_array[0,2], midi_array[0,0]], axis=0)

    end = midi_array[-1,0] + midi_array[-1,3]
    if end != N_bars*4:
        midi_array=np.insert(midi_array, -1, [end, 0, midi_array[-1,2], N_bars*4-end], axis=0)
        
    return midi_array


# TODO: PROVIDE REPETITION KEY
def convert_to_vector(midi_array, frame_factor, M):
    """
    Flattens a filled midi_array's midi numbers with encoding

        Parameters:
        -----------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration]]
            frame_factor (int): ratio of a single beat's length for pYIN frames,
                                due to the formulation, frame_factor samples in the pitch tracks is
                                worth a quarter beat. 
            M (int): decimation rate applied to the midi number sequence previously.

        Returns:
        --------
            symbolic_representation (ndarray): flat vector of encoded midi number sequence  
    """
    
    beat_factor =  4 * (frame_factor // M) #samples corresponds to 1 beat 

    symbolic_representation = []
    for midi in midi_array:

        note = int(midi[1])
        duration = int(midi[3]*beat_factor)

        symbolic_representation += [note] + [100]*(duration-1)
        
    return np.array(symbolic_representation)


def decode_NN_output(midi_code, frame_factor, velocity=120):
    """
    Converts the NN symbolic representation to a MIDI array
    """

    midi_code = unpack_repetitions(midi_code)

    # Zero midi numbers will be taken care of
    midi_array = midi_number_to_midi_array(midi_code, frame_factor, velocity=120)
    
    return midi_array


def unpack_repetitions(code):

    repetition_regions = calcRegionBounds(code==100)
    
    for l, u in repetition_regions:   
        code[l:u] = code[l-1]

    return code
    

def distance_to_C(key):
    """
    Returns the number of intervals between the Key and C.
    """

    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    return pitches.index(key)


def transpose_to_C(midi_array, key):

    N_intervals = distance_to_C(key)
    
    midi_array_T = midi_array.copy()
    
    MIDI_numbers = midi_array_T[:,1].astype(int)
    
    MIDI_numbers_T = [number-N_intervals for number in MIDI_numbers]
    
    midi_array_T[:,1] = MIDI_numbers_T
    
    return midi_array_T