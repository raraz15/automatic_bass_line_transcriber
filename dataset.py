import os
import sys
import json, traceback
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#sys.path.insert(0, '/mnt/d/projects/bassline_transcription')
sys.path.insert(0, '/scratch/users/udemir15/ELEC491/bassline_transcription')

from utilities import *
from bassline_transcriber.transcription import transpose_to_C

#ORIGINAL_MAX = 60 # C1 to C3
#ORIGINAL_MIN = 35 

MAX_NOTE = 51 
MIN_NOTE = 28 # result of transposition

SILENCE_CODE = 0
SUSTAIN_CODE = 100


def create_dataframes(track_dicts, bad_titles, M, directories):
    
    track_titles = track_dicts.keys()
    
    error_counter, note_filtered_counter, beat_f0_filtered_counter = 0, 0, 0
    
    codebook_pre, codebook_after, codebook_filtered = set(), set(), set()
    
    minor_matrix, major_matrix = [], []
    minor_titles, major_titles = [], []
    for title in track_titles:

        try:
            vector = load_symbolic_representation(title, directories, M) # already transposed to C
            
            if name_filter(title, bad_titles):
            
                if note_filter(vector):

                    codebook_pre = codebook_pre.union(set(vector))

                    vector = make_consecutive_codes(vector) # make the codes consecutive
                    codebook_after = codebook_after.union(set(vector))

                    scale_type = track_dicts[title]['Key'].split(' ')[-1]           

                    if scale_type == 'min':
                        minor_matrix.append(vector)
                        minor_titles.append(title)

                    elif scale_type == 'maj':
                        major_matrix.append(vector)
                        major_titles.append(title)
                else:
                    note_filtered_counter += 1
                    codebook_filtered = codebook_filtered.union(set(vector))
                    
            else:
                beat_f0_filtered_counter += 1
                
        except FileNotFoundError:
            error_counter += 1
        except KeyboardInterrupt:
            sys.exit(0)       
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            
    print('\nCodebook before correction:\n{}'.format(codebook_pre))
    print('\nCodebook after correction:\n{}'.format(codebook_after))
    print('\nFiltered Codes:\n{}'.format(sorted(codebook_filtered.difference(codebook_pre))))
            
    print('\n{}/{} Tracks couldnt found!'.format(error_counter, len(track_titles)))
    print('\n{}/{} Tracks filtered out because of notes!'.format(note_filtered_counter, len(track_titles)))
    print('\n{}/{} Tracks filtered out because of beat, f0 analysis!\n'.format(beat_f0_filtered_counter, len(track_titles)))

    major_matrix = np.stack(major_matrix, axis=0)
    minor_matrix = np.stack(minor_matrix, axis=0)

    df_minor = pd.DataFrame(minor_matrix)
    df_major = pd.DataFrame(major_matrix)

    df_minor['Title'] = minor_titles
    df_major['Title'] = major_titles

    df_minor = df_minor.reindex(columns=['Title'] + [x for x in np.arange(len(vector))])
    df_major = df_major.reindex(columns=['Title'] + [x for x in np.arange(len(vector))])
    
    print('Final Minor Dataset size: {}'.format(len(df_minor)))
    print('Final Major Dataset size: {}'.format(len(df_major)))
    
    for i in np.diff(list(codebook_after)):
        if i > 1:
            print('\nNonconsecutive codes!!!!')
        
    return df_minor, df_major

def df_from_codes(representations, track_titles, sustain=100, silence=0, MAX_NOTE=51, MIN_NOTE=28):
    
    X = representations.copy() # copy to avoid mistakes
    
    error_counter, note_filtered_counter = 0, 0
    codebook_pre, codebook_after, codebook_filtered = set(), set(), set()
    
    matrix, titles = [], []
    for vector, title in zip(X, track_titles):
    
        try:
                                   
            if note_filter(vector, sustain=sustain, silence=silence, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE):

                codebook_pre = codebook_pre.union(set(vector)) 

                vector = make_consecutive_codes(vector, sustain=sustain, silence=silence,
                                                MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE) # make the codes consecutive

                codebook_after = codebook_after.union(set(vector))

                matrix.append(vector)
                titles.append(title)

            else:                
                note_filtered_counter += 1
                codebook_filtered = codebook_filtered.union(set(vector))
                
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            
    print('\nCodebook before correction:\n{}'.format(sorted(codebook_pre)))
    print('\nCodebook after correction:\n{}'.format(sorted(codebook_after)))
    print('\nFiltered Codes:\n{}'.format(sorted(codebook_filtered.difference(codebook_pre))))
            
    print('\n{}/{} Tracks filtered out because of its notes!'.format(note_filtered_counter, len(track_titles)))
    
    matrix = np.stack(matrix, axis=0)
    df = pd.DataFrame(matrix)
    df['Title'] = titles
    df = df.reindex(columns=['Title'] + [x for x in np.arange(len(vector))])
    
    print('Final Dataset size: {}'.format(len(df)))
    
    for i in np.diff(list(codebook_after)):
        if i > 1:
            print('\nNonconsecutive codes!!!!')
        
    return df


def note_filter(vector, sustain=100, silence=0, MAX_NOTE=51, MIN_NOTE=28):
    """Check if the code contains unwanted notes."""
    flag = True
    for code in vector:
        if code != sustain and code != silence:
            if (code > MAX_NOTE) or (code < MIN_NOTE):              
                flag=False
                break
    return flag

def make_consecutive_codes(codes, sustain=100, silence=0, MAX_NOTE=51, MIN_NOTE=28):
    """Make the codes consecutive consecutive"""
    
    X = codes.copy()
    
    X[X==sustain] = MAX_NOTE+1
    X[X!=silence] -= MIN_NOTE-1
    
    return X

def name_filter(title, bad_titles):
    flag = True
    if title in bad_titles:
        flag = False
    return flag

def merge_track_dicts(*args):
    track_dicts = {}
    for dct in args:
        track_dicts = {**track_dicts, **dct}
    return track_dicts

# --------------------- Augmentation --------------------------

def control_silence(representation):
    """Check if the first bar is complete silence."""
    flag=True
        
    if (representation[:16] == np.array([0] + [100]*15)).all():        
        flag=False
    
    return flag

def repeat_dataset(df_titles, track_dicts, N_bars_repeat, directories):
     # how many bars to repeat

    N_bars = 4 # bassline length
    segment_length = N_bars_repeat*(512//N_bars)
    print('Segment length: {}'.format(segment_length))

    representations, bar_titles = [], []
    for title in df_titles:

        try:
            F0 = load_quantized_pitch_track(title, directories)[1]
            key = track_dicts[title]['Key'].split(' ')[0]

            for i in range(N_bars//N_bars_repeat):
                segment = F0[i*segment_length: (i+1)*segment_length]

                representation = bars_to_representation(segment, M, N_bars_repeat, key)

                if control_silence(representation):
                    representations.append(representation)
                    bar_titles.append(title) # duplicate for title column                
                else:
                    print('full silence')

        except KeyboardInterrupt:
            sys.exit(0)       
        except KeyError:
            pass
        except IndexError:
            pass
            #print(segment.shape)
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))

    representations = np.stack(representations, axis=0)
    bar_titles = np.stack(bar_titles, axis=0)
    print('Before concat: {}'.format(representations.shape))
    representations = np.concatenate([representations]*(N_bars//N_bars_repeat), axis=1)
    print('After concat: {}'.format(representations.shape))
    
    return representations, bar_titles

# ------------------------- Analysis ---------------------------------------

def count_keys(df, merged_track_dicts):
    
    counter = Counter()
    for title in df['Title']:

        track = merged_track_dicts[title]
        key, scale_type = track['Key'].split(' ')        
        counter[key] += 1

    return dict(sorted(counter.items(), key=lambda x: x[0].lower()))   

def count_notes(track_dicts, directories, M):
    
    note_counter, note_counter_T = Counter(), Counter()
    for title, track in track_dicts.items():

        try:    
            midi_array = load_bassline_midi_array(title, directories, M)
            midi_notes = midi_array[:,1].astype(int)
            
            key, scale_type = track['Key'].split(' ')
            
            midi_array_T = transpose_to_C(midi_array, key)
                        
            midi_notes_T = midi_array_T[:,1].astype(int)
            
            for note in midi_notes:
                note_counter[note] += 1

            for note in midi_notes_T: 
                note_counter_T[note] += 1
        except KeyboardInterrupt:
            sys.exit(0)
        except FileNotFoundError:
            pass
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))

    note_counter = dict(sorted(note_counter.items(), key=lambda x: x[0]))
    note_counter_T = dict(sorted(note_counter_T.items(), key=lambda x: x[0]))    
    
    return note_counter, note_counter_T

# ------------------------- PLotting ----------------------------------------------
def key_pie_charts(m_counter, M_counter):

    fig, ax = plt.subplots(ncols=2, figsize=(16,7), constrained_layout=True)
    
    values = np.array(list(m_counter.values()))
    explode = np.zeros(12)
    explode[np.argpartition(values, -3)[-3:]] = [0.05, 0.1, 0.2]

    ax[0].set_title('Minor Keys Bassline Transcriptions', fontsize=15)
    ax[0].pie(values, explode=explode, labels=m_counter.keys(),textprops={'fontsize': 15})
    ax[0].axis('equal')

    values = np.array(list(M_counter.values()))
    explode = np.zeros(12)
    explode[np.argpartition(values, -3)[-3:]] = [0.05, 0.1, 0.2]

    ax[1].set_title('Major Keys Bassline Transcriptions', fontsize=15)
    ax[1].pie(values, explode=explode, labels=M_counter.keys(),textprops={'fontsize': 15})
    ax[1].axis('equal')
    plt.show()
    

def plot_note_occurances(note_counter, note_counter_T, M):

    vals, labels = zip(*note_counter.items())
    vals_T, labels_T = zip(*note_counter_T.items())
    
    min_orig, max_orig = min(vals), max(vals)
    min_T, max_T = min(vals_T), max(vals_T)
    print('Original min: {}, max: {}'.format(min_orig, max_orig))
    print('Transposed min: {}, max: {}'.format(min_T, max_T))

    #pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    #p = ['B#0'] + [pitch+'1' for pitch in pitches] + [pitch+'2' for pitch in pitches] + ['C3']

    fig, ax = plt.subplots(nrows=2, figsize=(20,10), sharex=True)
    fig.suptitle('MIDI Number Occurance Frequencies (M={})'.format(M), fontsize=22)

    ax[0].set_title('Without Transposing', fontsize=18)
    ax[0].set_xlim([min_orig-1, max_orig+1])
    ax[0].bar(vals, labels)
    ax[0].tick_params(axis='both', which='major', labelsize=15)

    ax[1].set_title('After Transposing', fontsize=18)
    ax[1].set_xlim([min_T-1, max_T+1])
    ax[1].bar(vals_T, labels_T)
    ax[1].vlines([30, 48], 0, 25000, color='r', linewidth=5)
    ax[1].vlines([28, 51], 0, 25000, color='g', linewidth=5)
    ax[1].tick_params(axis='both', which='major', labelsize=15)

    #plt.savefig('MIDI_number_distribution.jpg')
    plt.show()