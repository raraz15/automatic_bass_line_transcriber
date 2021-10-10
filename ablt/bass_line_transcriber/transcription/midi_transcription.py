#!/usr/bin/env python
# coding: utf-8

import numpy as np

# TODO REPLACE zeros prior for warning handling.
def frequency_to_midi_sequence(F0, silence_code=0):
    """
    Maps a frequency array to midi numbers with silence regions indicated by the silence_code and ensures length.

        Parameters:
        -----------
            F0 (ndarray): frequency array
            silence_code (int, default=0): A code representing silences

        Returns:
        --------
            midi_seq (ndarray): numpy array of midi numbers, silence=silence_code
    """

    midi_seq = 12*np.log2(F0/440) + 69

    midi_seq = np.rint(np.nan_to_num(midi_seq, neginf=silence_code)).astype(int)

    return midi_seq


def midi_sequence_to_midi_array(midi_seq, M, N_qb=8, silence_code=0, velocity=120):
    """
    Downsamples and extracts onset, note, velocity and note length information from a sequence of midi picthes.

        Parameters:
        -----------
            midi_seq (ndarray): midi number sequence
            M (int): decimation rate between 1 and N_qb
            N_qb (int, default=8): number of samples a quarterbeat gets
            silence_code (int, default=0): A code int representing silences
            velocity (int, default=120): velocity of a midi note 

        Returns:
        --------
            midi_array (ndarray): numpy array of [[start_beat, midi number, velocity, duration]]
    """

    midi_seq = downsample_midi_sequence(midi_seq, M=M, N_qb=N_qb)
    
    # Number of samples in the midi_seq corresponding to a beat
    hop_ratio = 4*(N_qb//M)

    # adjust the beginning and the end of the loop
    change_indices = np.where(np.diff(midi_seq) != 0)[0]
    change_indices = np.insert(change_indices, [0, len(change_indices)], [-1, len(midi_seq)-1])
    note_lengths = np.diff(change_indices) / hop_ratio  # normalize to beats

    midi_array = []
    for i, j in enumerate(change_indices[:-1]):
        start_idx = j+1
        note = midi_seq[start_idx]
        if note != silence_code:  # non-zero notes only
            midi_array.append([start_idx/hop_ratio, note, velocity, note_lengths[i]])

    return np.array(midi_array)


def downsample_midi_sequence(midi_seq, M,  N_qb=8):
    """
    Downsamples a given midi number sequence uniformly.

        Parameters:
        -----------
            midi_seq (ndarray): midi number sequence
            M (int): decimation rate between 1 and N_qb
            N_qb (int, default=8): number of samples a quarterbeat gets

        Returns:
        --------
            midi_seq_decimated (ndarray): downsampled number sequence
    """

    assert M <= N_qb and M >= 1, 'Decimation rate must be smaller than N_qb={} points (quarter beat length)'.format(
        N_qb)
    assert not N_qb % M, 'N_qb must be divisble by the decimation rate!'

    # Downsample
    midi_seq_decimated = midi_seq[np.arange(0, len(midi_seq), M, dtype=int)]

    return midi_seq_decimated