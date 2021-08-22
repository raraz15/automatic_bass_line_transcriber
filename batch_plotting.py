import os, sys, json

import numpy as np
import pandas as pd
#from tqdm import tqdm
sys.path.insert(0, '/scratch/users/udemir15/ELEC491/bassline_transcription')
#sys.path.insert(0, '/mnt/d/projects/bassline_transcription') 

import plotting as plot
from bassline_transcriber import transcription
from utilities import *
from signal_processing import *

np.set_printoptions(suppress=True)

if __name__ == '__main__':

    directories = get_directories('data/directories.json')
    track_dicts_name ='TechHouse_total_track_dicts.json'

    scales, track_dicts, track_titles = read_metadata(directories['extraction'], track_dicts_name)

    with open('/scratch/users/udemir15/ELEC491/bassline_transcription/data/metadata/train_misc_names.txt', 'r') as infile:
        train_misc_names = infile.read().split('\n')
    with open('/scratch/users/udemir15/ELEC491/bassline_transcription/data/metadata/val_misc_names.txt', 'r') as infile:
        val_misc_names = infile.read().split('\n')
    misc_names = train_misc_names+val_misc_names

    with open('/scratch/users/udemir15/ELEC491/bassline_transcription/data/metadata/4020_titles.txt', 'r') as infile:
        titles = infile.read().split('\n')

    for title in misc_names:
        
        #if title not in misc_names:

        try:

            BPM = float(track_dicts[title]['BPM'])
            beat_length = 60/BPM
            track_scale = get_track_scale(title, track_dicts, scales)

            fs = 44100
            _, bassline = load_chorus_and_bassline(title, directories)

            quantized_pitch_track = load_quantized_pitch_track(title, directories)

            bassline_notes, unk_bassline_notes = transcription.extract_note_dicts(quantized_pitch_track, track_scale)

            center = True
            n_fft  = 4096

            spectrogram_beat_factor = 8
            win_length = int((beat_length/spectrogram_beat_factor)*fs) 
            hop_length = int(win_length/4) 

            bassline_spectrogram = extract_dB_spectrogram(bassline, n_fft, win_length, hop_length, center=center)

            plot.batch_plotting(title, directories,
                                bassline, bassline_spectrogram,
                                fs, hop_length,
                                F0_estimate=quantized_pitch_track,
                                plot_title='Misc',
                                save=True, figsize=(12,8))

        except Exception as ex:
            print(ex)