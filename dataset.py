import os, sys
import traceback
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import utilities
import bassline_transcriber.transcription as transcription
from representation import transpose_to_C

#ORIGINAL_MAX = 48 # C3 (middle C = C4 during transcription)
#ORIGINAL_MIN = 23 # B1 

#SILENCE_CODE = 0
#SUSTAIN_CODE = 100

# --------------------------------------- Dataframe Creation ---------------------------------------------------

def create_midi_sequences_array(track_titles, track_dicts, directories, silence_code=0):
    midi_sequences, valid_titles, keys, scales = [], [], [], []
    for title in track_titles:
        try:
            key, scale_type = track_dicts[title]['Key'].split(' ')
            pitch_track = utilities.load_quantized_pitch_track(title, directories)[1]
            midi_sequence = transcription.frequency_to_midi_sequence(pitch_track, silence_code=silence_code)
            midi_sequences.append(midi_sequence)
            valid_titles.append(title)
            keys.append(key)
            scales.append(scale_type)
        except KeyboardInterrupt:
            sys.exit(0)  
        except FileNotFoundError:
            pass
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
    midi_sequences = np.array(midi_sequences).reshape(-1, 512)
    print('There are {} midi sequences.'.format(midi_sequences.shape[0]))
    return midi_sequences, valid_titles, keys, scales


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


def merge_track_dicts(*args):
    track_dicts = {}
    for dct in args:
        track_dicts = {**track_dicts, **dct}
    return track_dicts


# TODO: MAKE THEM WORK
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
            F0 = utilities.load_quantized_pitch_track(title, directories)[1]
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

def count_keys(df, track_dicts):
    counter = Counter()
    for title in df['Title']:
        track = track_dicts[title]
        key = track['Key'].split(' ')[0]  
        counter[key] += 1
    return dict(sorted(counter.items(), key=lambda x: x[0].lower()))   

def count_notes(df):
    X = df.iloc[:, 3:].to_numpy()
    note_counter= Counter()
    for i, midi_sequence in enumerate(X):
        try:    
            for note in midi_sequence:
                note_counter[note] += 1
        except KeyboardInterrupt:
            sys.exit(0)
        except FileNotFoundError:
            pass
        except Exception as ex:
            print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))
    #del note_counter[0]
    note_counter = dict(sorted(note_counter.items(), key=lambda x: x[0]))
    return note_counter

def count_notes_with_transposing(df):
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