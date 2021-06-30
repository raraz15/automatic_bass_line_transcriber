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
import bassline_transcriber.transcription as transcription
from representation import encode_midi_sequence, transpose_to_C

#ORIGINAL_MAX = 60 # C3
#ORIGINAL_MIN = 35 # B1 

# MAX_NOTE = 51 # result of transposition
# MIN_NOTE = 28 

SILENCE_CODE = 0
SUSTAIN_CODE = 100

def load_data(data_params):     
    dataset_path, scale_type, M = data_params['dataset_path'], data_params['scale_type'], data_params['M']    
    dataset_name = data_params['dataset_name'] +'_{}_M{}.csv'.format(scale_type, M)
    dataset_dir = os.path.join(dataset_path, dataset_name)
    df = pd.read_csv(dataset_dir, header=None)
    titles = df[0].tolist()
    X = df[df.columns[1:]].to_numpy()    
    return X, titles

def append_SOS(X, SOS_token=-1):
    X = np.concatenate( (SOS_token*np.ones((X.shape[0],1), dtype=np.int64), X), axis=1)    
    return X+1 

def make_dataframe(X, titles, keys, scales):
    df = pd.DataFrame(X)
    df['Title'] = titles
    df['Key'] = keys
    df['Scale'] = scales
    df = df.reindex(columns=['Title']+['Key']+['Scale']+[x for x in np.arange(X.shape[1])])
    return df


def bars_to_representation(bar, M, N_bars, key):
        
    midi_array = transcription.frequency_to_midi_array(bar, M, N_bars, silence_code=0)
      
    representation = transcription.encode_midi_array(midi_array, M, N_bars, key, silence_code=0, sustain_code=100)
    
    return representation



# TODO: REPLACE WITH NEW VERSIONS
def create_dataframes(track_dicts, bad_titles, M, directories, sustain=100, silence=0, MAX_NOTE=51, MIN_NOTE=28):
    
    track_titles = track_dicts.keys()
    
    error_counter, note_filtered_counter, beat_f0_filtered_counter = 0, 0, 0
    
    codebook_pre, codebook_after, codebook_filtered = set(), set(), set()
    
    minor_matrix, major_matrix = [], []
    minor_titles, major_titles = [], []
    for title in track_titles:

        try:
            
            if title not in bad_titles:

                pitch_track = load_quantized_pitch_track(title, directories)[1]
                midi_sequence = transcription.frequency_to_midi_sequence(pitch_track, silence_code=silence)
                code = encode_midi_sequence(midi_sequence, M,
                                            sustain_code=sustain, silence_code=silence,
                                            MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE,
                                            key=track_dicts[title]['Key'].split(' ')[0]) # transposition inside
            
                if note_filter(code, sustain=sustain, silence=silence, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE):

                    codebook_pre = codebook_pre.union(set(code))

                    code = transcription.make_consecutive_codes(code, sustain=sustain, silence=silence, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE) # make the codes consecutive
                    codebook_after = codebook_after.union(set(code))

                    scale_type = track_dicts[title]['Key'].split(' ')[-1]           

                    if scale_type == 'min':
                        minor_matrix.append(code)
                        minor_titles.append(title)

                    elif scale_type == 'maj':
                        major_matrix.append(code)
                        major_titles.append(title)
                else:
                    note_filtered_counter += 1
                    codebook_filtered = codebook_filtered.union(set(code))
                    
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

    df_minor = df_minor.reindex(columns=['Title'] + [x for x in np.arange(len(code))])
    df_major = df_major.reindex(columns=['Title'] + [x for x in np.arange(len(code))])
    
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
    for code, title in zip(X, track_titles):
    
        try:
                                   
            if note_filter(code, sustain=sustain, silence=silence, MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE):

                codebook_pre = codebook_pre.union(set(code)) 

                code = transcription.make_consecutive_codes(code, sustain=sustain, silence=silence,
                                                MAX_NOTE=MAX_NOTE, MIN_NOTE=MIN_NOTE) # make the codes consecutive

                codebook_after = codebook_after.union(set(code))

                matrix.append(code)
                titles.append(title)

            else:                
                note_filtered_counter += 1
                codebook_filtered = codebook_filtered.union(set(code))
                
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
            
    print('\nCodebook before correction:\n{}'.format(sorted(codebook_pre)))
    print('\nCodebook after correction:\n{}'.format(sorted(codebook_after)))
    print('\nFiltered Codes:\n{}'.format(sorted(codebook_filtered.difference(codebook_pre))))
            
    if note_filtered_counter:
        print('\n{}/{} Tracks filtered out because of its notes!'.format(note_filtered_counter, len(track_titles)))
    
    matrix = np.stack(matrix, axis=0)
    df = pd.DataFrame(matrix)
    df['Title'] = titles
    df = df.reindex(columns=['Title'] + [x for x in np.arange(len(code))])
    
    print('Final Dataset size: {}'.format(len(df)))
    
    for i in np.diff(list(codebook_after)):
        if i > 1:
            print('\nNonconsecutive codes!!!!')
        
    return df

def code_filter(code, MAX_NOTE=51, MIN_NOTE=28):
    """Check if the code contains unwanted notes."""
    flag = True
    for symbol in code:
        if (symbol > MAX_NOTE-MIN_NOTE) or (symbol < 0):              
            flag=False
            break
    return flag

# TODO: REMOVE
def note_filter(code, sustain=100, silence=0, MAX_NOTE=51, MIN_NOTE=28):
    """Check if the code contains unwanted notes."""
    flag = True
    for symbol in code:
        if symbol != sustain and symbol != silence:
            if (symbol > MAX_NOTE) or (symbol < MIN_NOTE):              
                flag=False
                break
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

def repeat_dataset(df_titles, track_dicts, N_bars_repeat, directories, M):
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

def count_same_phrases(code, M, counter):
    
    code_re = code.reshape((4,4, 4*(8//M)))
    
    for b in range(4):

        bars = code_re[:,b,:] # beat b for all bars

        for i in range(4):
            for j in range(i+1,4):

                B0 = bars[i,:]
                B1 = bars[j,:]

                if np.array_equal(B0, B1):
                    key = '{}{}'.format(i,j)
                    counter[b][key] += 1
                    
    return counter 

def count_keys(df, track_dicts):
    counter = Counter()
    for title in df['Title']:
        track = track_dicts[title]
        key = track['Key'].split(' ')[0]  
        counter[key] += 1
    return dict(sorted(counter.items(), key=lambda x: x[0].lower()))   

def count_notes(df):
    X = df.iloc[:, 3:].to_numpy()
    note_counter, note_counter_T = Counter(), Counter()
    for i, midi_sequence in enumerate(X):
        try:    
            midi_sequence_T = transpose_to_C(midi_sequence, df['Key'][i])
            for note in midi_sequence:
                note_counter[note] += 1
            for note in midi_sequence_T: 
                note_counter_T[note] += 1
        except KeyboardInterrupt:
            sys.exit(0)
        except FileNotFoundError:
            pass
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
    del note_counter[0]
    del note_counter_T[0]
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
    

def plot_note_occurances(note_counter, note_counter_T, M, title=''):

    vals, labels = zip(*note_counter.items())
    vals_T, labels_T = zip(*note_counter_T.items())
    
    min_orig, max_orig = min(vals), max(vals)
    min_T, max_T = min(vals_T), max(vals_T)
    print('Original min: {}, max: {}'.format(min_orig, max_orig))
    print('Transposed min: {}, max: {}'.format(min_T, max_T))

    fig, ax = plt.subplots(nrows=2, figsize=(20,10), sharex=True)
    fig.suptitle('MIDI Number Occurance Frequencies (M={})'.format(M), fontsize=22)

    ax[0].set_title('Without Transposing', fontsize=18)
    ax[0].set_xlim([min_orig-1, max_orig+1])
    ax[0].bar(vals, labels)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].vlines([24, 48], 0, 100000, color='r', linewidth=5)
    ax[0].xaxis.set_tick_params(labelbottom=True)

    ax[1].set_title('After Transposing', fontsize=18)
    ax[1].set_xlim([min_T-1, max_T+1])
    ax[1].bar(vals_T, labels_T)
    ax[1].vlines([17, 36], 0, 100000, color='r', linewidth=5)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    if title:
        plt.savefig('MIDI_number_distribution.jpg')
    plt.show()