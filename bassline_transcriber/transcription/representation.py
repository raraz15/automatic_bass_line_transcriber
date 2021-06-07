#!/usr/bin/env python
# coding: utf-8

import numpy as np

from utilities import calcRegionBounds
from .midi_transcription import midi_number_to_midi_array
from MIDI_output import create_MIDI_file

# -------------------------------------------- ENCODING -------------------------------------------------------------------------

def encode_midi_array(midi_array, M, N_bars, key=None, N_qb=8, silence_code=0, sustain_code=100):
    """Encodes a given midi array """

    if key is not None:
        midi_array = transpose_to_C(midi_array, key)

    filled_midi_array = fill_midi_array(midi_array, N_bars, silence_code)

    symbolic_representation = convert_to_vector(filled_midi_array, N_qb, M, sustain_code)

    return symbolic_representation


def fill_midi_array(midi_array, N_bars, silence_code=0):
    """
    Fills the midi_array silence indices and corrects the endpoints for encoding stage.
        
        Parameters:
        -----------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration]]
            N_bars (int): Number of bars that the midi_array corresponds to
            silence_code (int, default=0): code corresponding to a silence instance

        Returns:
        --------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration]]
    """
    
    indices, rows = [], [] # indices for silence insertion and the notes to insert

    for idx, midi in enumerate(midi_array[:-1]):

        end = midi[0] + midi[3] # start_idx + duration

        if end != midi_array[idx+1, 0]: # if there is a gap between the next note
            indices.append(idx+1) 
            rows.append([end, silence_code, 120, midi_array[idx+1, 0]-end]) # fill the gap with 0

    if len(indices) > 0: # if there are gaps, fill
        midi_array = np.insert(midi_array, indices, rows, axis=0)


    # Ensure beginning and end indices
    if midi_array[0,0] != 0.0:
        midi_array=np.insert(midi_array, 0, [0.0, silence_code, midi_array[0,2], midi_array[0,0]], axis=0)

    end = midi_array[-1,0] + midi_array[-1,3]
    if end != N_bars*4:
        midi_array=np.insert(midi_array, -1, [end, silence_code, midi_array[-1,2], N_bars*4-end], axis=0)
        
    return midi_array


def convert_to_vector(midi_array, frame_factor, M, sustain_code=100):
    """
    Flattens a filled midi_array's midi numbers after encoding

        Parameters:
        -----------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration],]
            frame_factor (int): ratio of a single beat's length for pYIN frames,
                                due to the formulation, frame_factor samples in the pitch tracks is
                                worth a quarter beat. 
            M (int): decimation rate applied to the midi number sequence previously.
            sustain_code(int, default=100): sustain code 

        Returns:
        --------
            symbolic_representation (ndarray): flat vector of encoded midi number sequence  
    """
    
    beat_factor =  4 * (frame_factor // M) #samples corresponds to 1 beat 

    symbolic_representation = []
    for midi in midi_array:

        note = int(midi[1])
        duration = int(midi[3]*beat_factor)

        symbolic_representation += [note] + [sustain_code]*(duration-1)
        
    return np.array(symbolic_representation)
    

def distance_to_C(key):
    """Returns the number of intervals between the Key and C."""

    pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitches.index(key)


def transpose_to_C(midi_array, key):
    """Transposes a given midi array to C by calculating root distances."""

    N_intervals = distance_to_C(key) # root notes distance to C in semitones
    
    midi_array_T = midi_array.copy()
    MIDI_numbers = midi_array_T[:,1].astype(int)
    
    MIDI_numbers_T = [m-N_intervals for m in MIDI_numbers] # transpose each midi number
    
    midi_array_T[:,1] = MIDI_numbers_T
    
    return midi_array_T


# TODO TRANSPOSE INISDE
def encode_midi_array_new(midi_array, N_qb, M, N_bars, silence_code=0, sustain_silence=-1, sustain_note=-2):

    filled_midi_array = fill_midi_array(midi_array, N_bars, silence_code)

    symbolic_representation = convert_to_vector_new(filled_midi_array, N_qb, M, silence_code, sustain_silence, sustain_note)

    return symbolic_representation


def convert_to_vector_new(midi_array, frame_factor, M, silence_code=0, sustain_silence=-1, sustain_note=-2):
    """
    Flattens a filled midi_array's midi numbers after encoding

        Parameters:
        -----------
            midi_array (ndarray): midi note array of [[start_idx, midi_number, velocity, duration],]
            frame_factor (int): ratio of a single beat's length for pYIN frames,
                                due to the formulation, frame_factor samples in the pitch tracks is
                                worth a quarter beat. 
            M (int): decimation rate applied to the midi number sequence previously.
            silence_code (default = 0)
            sustain(int, default=100): sustain symbol

        Returns:
        --------
            symbolic_representation (ndarray): flat vector of encoded midi number sequence  
    """
    
    beat_factor =  4 * (frame_factor // M) #samples corresponds to 1 beat 

    symbolic_representation = []
    for midi in midi_array:

        note = int(midi[1])
        duration = int(midi[3]*beat_factor)

        if note != silence_code:
            symbolic_representation += [note] + [sustain_note]*(duration-1)
        else:
            symbolic_representation += [note] + [sustain_silence]*(duration-1)
        
    return np.array(symbolic_representation)


# --------------------------------------------------- DECODING ------------------------------------------------

def NN_output_to_midi_array(representation, N_qb, frame_factor, min_note=28, silence_code=0, sustain_code=100, velocity=120):
    """
    Converts the NN symbolic representation to a MIDI array
    """
    midi_code = representation_to_code(representation, min_note=min_note, silence_code=silence_code, sustain_code=sustain_code)

    midi_code = unpack_repetitions(midi_code, sustain_code)

    # Silence code will be taken care of
    midi_array = midi_number_to_midi_array(midi_code, N_qb, frame_factor, silence_code=silence_code, velocity=velocity)
    
    return midi_array


def representation_to_code(representation, min_note=28, silence_code=0, sustain_code=100):
    """Converts NN output (consecutive integers) to midi code  0, 28,29,..., 51, 100  for example"""

    code = representation.copy()
    
    code[code!=silence_code] += min_note-1

    max_code = code.max()
    
    code[code==max_code] = sustain_code # max will be the sustain
        
    return code


def unpack_repetitions(code, sustain_code=100):

    """Converts sustain codes to the note itself."""
    repetition_regions = calcRegionBounds(code==sustain_code)
    
    for l, u in repetition_regions:   
        code[l:u] = code[l-1]

    return code


def NN_output_to_MIDI_file(representation, BPM, title, output_dir, 
                            N_qb, frame_factor, middle_c='C3', tpb=960*16,
                            min_note=28, silence_code=0, sustain_code=100, velocity=120):

    midi_array = NN_output_to_midi_array(representation, N_qb=N_qb, frame_factor=frame_factor, min_note=min_note,
                            silence_code=silence_code, sustain_code=sustain_code, velocity=velocity)

    create_MIDI_file(midi_array, BPM, title, output_dir, middle_c=middle_c, tpb=tpb)