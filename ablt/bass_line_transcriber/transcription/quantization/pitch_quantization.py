#!/usr/bin/env python
# coding: utf-8

from collections import Counter

import numpy as np

from ....utilities import sample_and_hold
from ....constants import SUB_BASS_FREQUENCIES

# TODO: equal votes in majority voting

def uniform_quantization(pitch_track, segments, epsilon=2):
    """
    Uniformly quantizes each given segment independently.

        Parameters:
        -----------

            pitch_track (tupple): (time_axis, F0) where both are np.ndarray
            segments(tupple): voiced region (boundaries, lengths, indices)
            epsilon (int, default=4): freq_bound = delta_scale/epsilon determines if quantization will happen.
        
        Returns:
        --------
            
            pitch_track_quantized (tupple): (time_axis, quantized_F0) where both are np.ndarray

    """

    boundaries, lengths, indices = segments

    # Form the Pitch Histogram and Do Majority Voting for each region independently
    pitch_histograms = create_pitch_histograms(pitch_track[1], boundaries, epsilon)
    majority_pitches = get_majority_pitches(pitch_histograms)    

    #Quantize Each Region
    quantized_non_zero_frequencies = sample_and_hold(majority_pitches,  lengths)
    assert len(indices) == len(quantized_non_zero_frequencies), 'Hold lengths do not match'

    # replace regions with quantized versions
    quantized_pitches = pitch_track[1].copy()
    np.put(quantized_pitches, indices, quantized_non_zero_frequencies)

    pitch_track_quantized = (pitch_track[0], quantized_pitches) # (time, freq) 

    return  pitch_track_quantized


def quantize_frequency(f, epsilon):
    """
    Using epsilon balls around the MIDI pitch frequencies, quantizes a given frequency.
    
    Parameters:
    -----------

        f (float): frequency in Hz.
        epsilon (int): freq_bound = delta_scale/epsilon determines if quantization will happen.

    Returns:
    --------

        f (float): quantized frequency in hertz
        
    """
    
    if f: # for non zero frequencies
                
        delta_array = np.abs(f - np.array(SUB_BASS_FREQUENCIES)) # distances to the notes of the scale
        delta_min = np.min(delta_array) # smallest such distance

        delta_bound = np.min(np.diff(SUB_BASS_FREQUENCIES)) / epsilon 

        if delta_min <= delta_bound: # if there is a note closeby
            note_idx = np.where(delta_array==delta_min)[0][0] # index of the corresponding note in the scale
            f = SUB_BASS_FREQUENCIES[note_idx] # quantize pitch
            
    return f 


def single_pitch_histogram(F0, epsilon):
    """
    Creates a single pitch histogram for a given interval by quantizing each frequency.

    Parameters:
    -----------

        F0 (array): F0 array.
        epsilon (int): freq_bound = delta_scale/epsilon determines if quantization will happen.

    Returns:
    --------

        pitch_histogram (Counter): A Counter histogram of all the frequencies in the interval.

    """
   
    return Counter([quantize_frequency(f, epsilon) for f in F0])


def create_pitch_histograms(F0, boundaries, epsilon=2):  
    """
    For each time interval, quantizes the frequencies and creates a pitch histogram.
    
    Parameters:
    -----------

        F0_estimate (array): freq
        boundaries: (int or np.ndarray) Number of samples each time interval has. 
                    6 samples correspond to 1/8th beat and 12 1/4th for 120<=BPM<=130.
                    you can also provide the boundary of each region separately in an ndarray
        epsilon (int): freq_bound for frequency quantization. default = 2
                    
    Returns:
    --------

        pitch_histograms: (list) a list of pitch histogram Counters()

    """
    
    assert (isinstance(boundaries, int) or isinstance(boundaries, np.ndarray)), \
         ('provide a single interval length or an ndarray of voiced region boundaries')
    
    if isinstance(boundaries, int): # create the boundaries with uniform interval length
        boundaries = [[i*boundaries, (i+1)*boundaries] for i in range(int(len(F0)/boundaries))]

    pitch_histograms = []        
    for start, end in boundaries:
        interval_pitch_histogram = single_pitch_histogram(F0[start:end], epsilon)
        pitch_histograms.append(interval_pitch_histogram)
            
    return pitch_histograms


def get_majority_pitches(chunk_dicts):
    """Takes the majority pitch in an interval's pitch histogram."""
    
    return [max(hist, key=lambda k: hist[k]) for hist in chunk_dicts]