#!/usr/bin/env python
# coding: utf-8

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa


# DIRECTORY INIT

def get_directories(path):
    """
    Read the json file that contains all the read and write directories.
    """

    with open(path, 'r') as infile:
        directories = json.load(infile)

    return directories

def init_folders(directories):
    """
    Creates all the folders specified in the directories dict.
    """
    
    for directory in directories.values():

        if isinstance(directory, str):
            if not os.path.exists(directory):
                os.mkdir(directory)
        else:
            init_folders(directory)

            #for direct in directory.values():
            #    if not os.path.exists(direct):
            #        os.mkdir(direct)

#def init_exceptions_log(directories, date):
#    with open(os.path.join(directories['exceptions'], '{}.txt'.format(date)), 'w') as outfile:
#        outfile.write('')        

# METADATA

def get_track_scale(title, track_dicts, scales):
    """
    Finds the scale of a track from the track dicts and creates track_scale tupple.

        Parameters:
        -----------
            title (str): title of the track
            track_dicts (dict): a dict of track dicts
            scales (dict): the dict containing the frequency information for all the scales

        Returns:
        --------
            track_scale (tupple): (notes, scale_frequencies, out_notes, out_frequencies) where:
                            notes: note_name+octave e.g. C0 or F#2
                            scale_frequencies: list of 2 octaves of frequencies
                            out_notes: note names of notes that are not in the scale
                            out_frequencies: corresponding frequencies
    """
    
    key, scale_type = track_dicts[title]['Key'].split(' ')
    scale_frequencies = scales[key][scale_type]['frequencies']
    notes = [note+'0' for note in scales[key][scale_type]['notes']]
    notes += [note+'1' for note in scales[key][scale_type]['notes']]

    scale_frequencies = scale_frequencies['0'] + scale_frequencies['1'] 

    out_notes = [note+'0' for note in scales[key][scale_type]['out_of_scale']['notes']]
    out_notes += [note+'1' for note in scales[key][scale_type]['out_of_scale']['notes']]
    out_frequencies = scales[key][scale_type]['out_of_scale']['frequencies']

    track_scale = (notes, scale_frequencies, out_notes, out_frequencies)
        
    return track_scale


def get_track_dicts(directories):

    with open(os.path.join(directories['metadata'], 'TechHouse_track_dicts.json'),'r') as infile:
        track_dicts = json.load(infile)

    return track_dicts, list(track_dicts.keys())


def read_metadata(directories):

    with open(os.path.join(directories['metadata'], 'scales_frequencies.json'), 'r') as infile:
        scales = json.load(infile)
        
    track_dicts, track_titles = get_track_dicts(directories)

    return scales, track_dicts, track_titles



# INSPECTION

def load_chorus_and_bassline(title, directories):
    """
    Loads experiment outputs from numpy arrays.
    """
    
    chorus = np.load(directories['extraction']['chorus']['chorus_array']+'/'+title+'.npy')
    bassline = np.load(directories['extraction']['bassline']+'/'+title+'.npy')  
    
    return chorus, bassline

def load_track(track_title, fs):
    """
    Loads a track given title.
    """
    return librosa.load(os.path.join(directories['extraction']['clip_dir'],track_title+'.mp3'), sr=fs)

def load_F0_estimate(title, directories):    
    return np.load(directories['transcription']['bassline_transcription']['F0_estimate']+'/{}.npy'.format(title))

def load_pitch_track(title, directories):   
    return np.load(directories['transcription']['bassline_transcription']['pitch_track']+'/{}.npy'.format(title))

def load_quantized_pitch_track(title, directories):     
    return np.load(directories['transcription']['bassline_transcription']['quantized_pitch_track']+'/{}.npy'.format(title))

def load_bassline_midi_array(title, directories, M):
    return np.load(directories['midi']['midi_array'][str(M)]+'/{}.npy'.format(title))

def load_symbolic_representation(title, directories, M):
    return np.load(directories['transcription']['symbolic_representation'][str(M)]+'/{}.npy'.format(title))

def load_numpy_midi(midi_dir, file_name):  
    return np.load(os.path.join(midi_dir, file_name))

def print_plot_play(x, Fs, text=''):
    
    print('%s\n' % (text))
    print('Fs = %d, x.shape = %s, x.dtype = %s' % (Fs, x.shape, x.dtype))
    plt.figure(figsize=(8, 2))
    plt.plot(x, color='gray')
    plt.xlim([0, x.shape[0]])
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    ipd.display(ipd.Audio(data=x, rate=Fs)) #,normalize=False

def inspect_audio_outputs(track_title, directories, fs=44100):
    chorus, bassline = load_chorus_and_bassline(track_title, directories)
    
    print('\t\t{}\n'.format(track_title))
    print_plot_play(chorus, fs, 'Chorus')
    print_plot_play(bassline, fs, 'Bassline')
    

def search_idx(query, track_titles):
    found = False
    for idx, title in enumerate(track_titles):
        if title == query:
            print(idx)
            found = True

    if not found:
        print("Track couldn't found!")



# Beat, frequency

def create_frequency_bins(fs, n_fft): 
    bin_width = (fs/2) / n_fft
    frequency_bins = np.arange(0, int((n_fft/2)+1))*bin_width
    return frequency_bins, bin_width


def get_chorus_beat_positions(title, directories):
    """
    Loads the beat positions of the progression.
    """
    return np.load(directories['extraction']['chorus']['chorus_beat_positions']+'/'+title+'.npy')

def get_beat_positions(title):
    """
    Loads the beat positions for the complete track.
    """ 
    return np.load(directories['extraction']['chorus']['beat_positions']+'/'+title+'.npy')
    
def get_bar_positions(beat_positions):
    """
   Finds the bar positions from a gşven beat positions array.     
    """    
    return np.array([val for idx,val in enumerate(beat_positions) if not idx%4])


def get_quarter_beat_positions(beat_positions):
    quarter_beats = []
    for i in range(len(beat_positions)-1):
        for qb in np.linspace(beat_positions[i],beat_positions[i+1], 4, endpoint=False):
            quarter_beats.append(qb)
            
    return np.array(quarter_beats)


def get_eighth_beat_positions(beat_positions):
    eighth_beats = []
    for i in range(len(beat_positions)-1):
        for qb in np.linspace(beat_positions[i],beat_positions[i+1], 8, endpoint=False):
            eighth_beats.append(qb)
            
    return eighth_beats


#  Miscallenous


def sample_and_hold(samples, N_samples):
    """
    Repeats each sample N_samples times correspondingly.
    """
    
    if isinstance(N_samples, int): # uniform sample rate
        return [f for f in samples for _ in range(N_samples)]
    
    else: # varying sample length 
        return [sample for idx, val in enumerate(N_samples) for sample in sample_and_hold([samples[idx]], val)] 



# Region

def calcRegionBounds(bool_array):
    '''
    Returns the lower and upper bounds of contiguous regions.
    Upper bound is not included in the region i.e [start, end)

    Parameters
    ==========
    bool_array  1-D Binary numpy array
    '''
    assert(bool_array.dtype == 'bool' )
    idx = np.diff(np.r_[0, bool_array, 0]).nonzero()[0]
    assert(len(idx)%2 == 0)
    return np.reshape(idx, (-1,2))



# Saving

def export_function(array, directory, title):

    export_path = os.path.join(directory, '{}.npy'.format(title))
    np.save(export_path, array)


# printing

def print_midi_array(midi_array):
    
    x = list([list(row) for row in midi_array])
    x.insert(0,['Start Beat', 'MIDI Number', 'Velocity', 'Duration'])

    print('{:^59}\n'.format('Bassline MIDI Array'))
    print('{:^15}{:^16}{:^14}{:^14}'.format('Start Beat', 'MIDI Number', 'Velocity', 'Duration'))
    print('-'*59)
    for row in midi_array:
        
        start, dur = row[0], row[3]
        m, vel = row[1].astype(int), row[2].astype(int)

        #print('| {:>7}     |{:>9}      |   {:>6}      |  {:>7}    |'.format(start, m, vel, dur))    
        print('|{:^13}|{:^15}|{:^13}|{:^13}|'.format(start, m, vel, dur))

def print_symbolic_representation(symbolic_representation):
     
    print('{:^66}\n'.format('Bassline Symbolic Representation'))
    print(symbolic_representation[np.arange(0, len(symbolic_representation)).reshape(4,-1)])
    print('\nRepresentation Vector Length: {} (= 4 Bars = 16 Beats = 64 QuarterBeats)'.format(len(symbolic_representation)))   