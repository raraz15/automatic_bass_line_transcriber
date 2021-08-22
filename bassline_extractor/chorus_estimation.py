#!/usr/bin/env python
# coding: utf-8

import numpy as np

from signal_processing import lp_and_normalize
from utilities import sample_and_hold, get_bar_positions, get_beat_positions


# TODO : WRITE AN ALGORITHM, REFER HERE
def drop_detection(track, beat_positions, fs, epsilon):
    """
    Detects drops of a track using beat positions.
        
        Parameters:
        -----------
            track (ndarray): audio track
            bar_positions (ndarray): array of bar positions (in time)
            fs (int): sampling rate
            epsilon (int, default=1): determines the threshold value considering a drop

        Returns:
        --------
            drop_beat_idx (ndarray): main drop's beat index 
            possible_drop_indices (ndarray): beat indices of all possible drops

    """

    bar_positions = get_bar_positions(beat_positions)

    track_lp = lp_and_normalize(track, 256, fs)

    bar_energies = calculate_bar_energies(track_lp, bar_positions, fs)
    cell_energies = calculate_mean_cell_energies(bar_energies)
    smoothed_cell_energies = sample_and_hold(cell_energies, 4)

    possible_drops = find_drops(smoothed_cell_energies, epsilon)
    estimated_drop = drop_picking(possible_drops)

    if not estimated_drop[0]==None: # first entry could be 0.0
        drop_beat_idx=np.array(estimated_drop[0]*4) # cell = 4*bar
    else:
        drop_beat_idx=0

    return drop_beat_idx, np.array([idx*4 for idx in possible_drops[0]])


def find_drops(energies, epsilon=1):
    """
    Returns possible drop indices defined as the first high energy idx after breakdown section.

        Parameters:
        -----------
            energies (ndarray): energy array
            epsilon (int, default=1): determines the threshold value considering a drop

        Returns:
        --------
            possible_drops (tupple): (drop_indices, drop_energies) where drop_indices correspond to cell_indices
    """
    
    average_energy = np.mean(energies)
    energy_deviation = np.std(energies)

    threshold = average_energy - (energy_deviation/epsilon)

    low_energy_indices = np.array([i for i, energy in enumerate(energies) if energy <= threshold] + [len(energies)])

    discontinuity_indices = np.where(np.diff(low_energy_indices) != 1)[0]

    energy_rise_indices = low_energy_indices[discontinuity_indices]

    drop_indices, drop_energies = [], []
    for idx in energy_rise_indices:
        
        e = energies[idx+1]
        
        if e > threshold:
            drop_indices.append(idx+1)
            drop_energies.append(e)
                
    possible_drops = (np.array(drop_indices), drop_energies)
    
    return possible_drops


# TODO: EXPLAIN ALGO
def drop_picking(possible_drops, threshold=1000):
    """
    Picks a drop between possible drops.

        Parameters:
        -----------
            possible_drops (tupple): (drop_indices, drop_energies), where both are ndarray
            threshold (int): a threshold for selecting between two prominant drops (HAS A MEANING)

        Returns:
        --------
            drop (tupple): (drop_idx, drop_energy): index of the cell where the drop happens and the corresponding energy

    """
    
    drop_indices, drop_energies = possible_drops
    
    if drop_energies:

        if len(drop_energies) < 2:
            
            drop_idx = drop_indices[0]
            drop_energy = drop_energies[0]

        else:

            if drop_energies[1] - drop_energies[0] >= threshold:
                drop_idx = drop_indices[1]
                drop_energy = drop_energies[1]            
            else:
                drop_idx = drop_indices[0]
                drop_energy = drop_energies[0]   

    else:
        print('No drop detected!')
        drop_idx, drop_energy = None, None

    drop = (drop_idx, drop_energy)
    return drop


def calculate_bar_energies(track, bar_positions, fs):
    """
    Calculates the energies of each bar in a griven track using the bar positions.

        Parameters:
        -----------
            track (ndarray): audio track
            bar_positions (ndarray): array of bar positions (in time)
            fs (int): sampling rate

        Returns:
        --------
            bar_energies (ndarray): energy of each bar
    """
   
    bar_energies = []
    for idx in range(len(bar_positions)-1):
        
        segment = track[int(fs*bar_positions[idx]):int(fs*bar_positions[idx+1])]

        bar_energies.append(np.sum(np.square(segment)))
        
    return bar_energies


def calculate_mean_cell_energies(bar_energies):
    """
    Calculates the mean energy for each cell (4 bars)
    """    
    cell_boundaries = np.arange(0, len(bar_energies)+1, 4)
        
    cell_energies = []
    for i, j in enumerate(cell_boundaries[:-1]):
        
        start_idx = j
        end_idx = cell_boundaries[i+1]
                    
        cell_energies.append(np.mean(bar_energies[start_idx:end_idx]))
        
    return cell_energies
    

def check_chorus_beat_grid(chorus_beat_positions, beat_length):

    rounded_beat_length = np.around(beat_length, 2)

    beat_length_estimations = np.diff(chorus_beat_positions)
    length_deviations = np.abs(rounded_beat_length - beat_length_estimations)

    large_deviation_indices = np.where(length_deviations > 0.011)[0]
    
    return large_deviation_indices