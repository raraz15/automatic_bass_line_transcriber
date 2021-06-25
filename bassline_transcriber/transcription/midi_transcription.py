#!/usr/bin/env python
# coding: utf-8

import numpy as np


def frequency_to_midi_sequence(F0, middle_c='C3', silence_code=0):
    """
    Maps a frequency array to midi notes with silence regions indicated by silence_code.

        Parameters:
        -----------
            F0 (ndarray): frequency array
            middle_c {'C3', 'C4'}: C3 for Ableton sonification.
            silence_code (int): A code int representing silences,default=0

        Returns:
        --------
            midii_number_seq (ndarray): a numpy array of midi numbers
    """
    assert middle_c in ['C3', 'C4'], 'Middle C must be C3 or C4!'

    if middle_c == 'C3':  # convert to midi
        midi_number_seq = 12*np.log2(F0/440) + 69 + 12
    else:
        midi_number_seq = 12*np.log2(F0/440) + 69

    # replace -inf with 0
    midi_number_seq = np.rint(np.nan_to_num(midi_number_seq, neginf=silence_code)).astype(int)

    return midi_number_seq


def downsample_midi_number_sequence(midi_number_seq, M,  N_qb=8, N_bars=4):
    """
    Downsamples a given midi number sequence uniformly.

        Parameters:
        -----------
            midi_number_seq (ndarray): midi number sequence
            M (int): decimation rate between 1 and N_qb
            N_qb (int, default=8): number of samples a quarterbeat gets
            N_bars (int, default=4): number of bars the section has.

        Returns:
        --------
            midi_number_seq_decimated (ndarray): downsampled number sequence
    """

    assert M <= N_qb and M >= 1, 'Decimation rate must be smaller than N_qb={} points (quarter beat length)'.format(
        N_qb)
    assert not N_qb % M, 'N_qb must be divisble by the decimation rate!'

    N_required = 16*N_bars*N_qb  # required input signal length

    if len(midi_number_seq) < N_required:  # pad if needed
        midi_number_seq = np.append(
            midi_number_seq, midi_number_seq[-(N_required-len(midi_number_seq)):])

    # Downsample
    midi_number_seq_decimated = midi_number_seq[np.arange(
        0, N_required, M, dtype=int)]

    return midi_number_seq_decimated


def midi_sequence_to_midi_array(midi_number_seq, M, N_qb=8, silence_code=0, velocity=120):
    """
    Extracts onset, note, velocity and note length information from a midi number sequence.
    The zero midi number will be considered silence.

        Parameters:
        -----------
            midi_number_seq (ndarray): midi number sequence
            M (int): decimation rate between 1 and N_qb
            N_qb (int, default=8): number of samples a quarterbeat gets
            velocity (int): velocity of a midi note 

        Returns:
        --------
            midi_array (ndarray): numpy array of [[start_beat, midi number, velocity, duration]]
    """
    
    # Number of samples in the midi_number_seq corresponding to a beat
    beat_factor = 4*(N_qb//M)

    # find where the notes change
    change_indices = np.where(np.diff(midi_number_seq) != 0)[0]

    # adjust the beginning and the end of the loop
    change_indices = np.insert(change_indices, [0, len(change_indices)], [-1, len(midi_number_seq)-1])

    note_lengths = np.diff(change_indices) / beat_factor  # normalize to beats

    midi_array = []
    for i, j in enumerate(change_indices[:-1]):

        start_idx = j+1
        note = midi_number_seq[start_idx]

        if note != silence_code:  # non-zero notes only
            midi_array.append([start_idx/beat_factor, note, velocity, note_lengths[i]])

    return np.array(midi_array)


def frequency_to_midi_array(F0, M, N_bars=4, N_qb=8, silence_code=0, velocity=120):
    """
    Creates a midi note array from a given pitch track frequency array. 

        Parameters: 
        -----------
            F0 (ndarray): frequency array           
            M (int): decimation rate between 1 and N_qb
            N_bars (int, default=4): number of bars in the section 
            N_qb (int, default=8): number of points a quarterbeat gets
            velocity (int, default=120): The velocity of the midi notes.
            silence (int, default=0): Silence code

        Returns:
        --------
            midi_array (ndarray): numpy array of [[start_beat, midi number, velocity, duration],]
    """

    midi_sequence = frequency_to_midi_sequence(F0, M, N_qb, N_bars, silence_code)

    # create the midi note array
    midi_array = midi_sequence_to_midi_array(midi_sequence, M, N_qb, silence_code=silence_code, velocity=velocity)

    return midi_array