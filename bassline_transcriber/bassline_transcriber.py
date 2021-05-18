#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from tqdm import tqdm
import warnings
import traceback

project_dir = '/mnt/d/projects/bassline_transcription'
sys.path.insert(0, project_dir)
warnings.filterwarnings('ignore') 

import numpy as np
from matplotlib.pyplot import close, clf # required for preventing memory leakge while plotting

from .transcriber_classes import BasslineTranscriber
from signal_processing import extract_dB_spectrogram
from plotting import waveform_and_note_spectrogram
from utilities import (get_directories, init_folders, init_exceptions_log,
                        read_metadata, export_function,get_chorus_beat_positions)


def transcribe_single_bassline(title, directories, scales, track_dicts, M, quantization_scheme,
                                filter_unk, epsilon, pYIN_threshold, plot, date=''):

    try:

        print('\n'+title)

        bassline_transcriber=BasslineTranscriber(title, directories, scales, track_dicts, M)

        bassline_transcriber.extract_pitch_track(pYIN_threshold)

        bassline_transcriber.quantize_pitch_track(filter_unk, epsilon, quantization_scheme)

        bassline_transcriber.extract_notes()

        bassline_transcriber.create_midi_array()

        bassline_transcriber.create_symbolic_representation()

        bassline_transcriber.create_midi_file()


        bassline_transcriber.export_F0_estimate()

        bassline_transcriber.export_pitch_track()

        bassline_transcriber.export_quantized_pitch_track()

        bassline_transcriber.export_midi_array()

        bassline_transcriber.export_symbolic_representation()


        if plot:
            plot_waveform_and_note_spectrogram(Bassline_Transcriber)

    except KeyboardInterrupt:
        sys.exit()
        pass
    #except UnboundLocalError:

    except Exception as ex:     
        print("There was an error on: {}".format(title))
        exception_str = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        print(exception_str+'\n')       
        with open(os.path.join(directories['transcription']['exceptions'], '{}.txt'.format(date)), 'a') as outfile:
            outfile.write(title+'\n'+exception_str+'\n\n')


def main(directories_path=project_dir, M=1, quantization_scheme='adaptive', filter_unk=True,
            epsilon=2, pYIN_threshold=0.05, plot=False):

    directories, scales, track_dicts, track_titles, date = prepare(directories_path)

    start_time = time.time()
    for title in tqdm(track_titles):

        transcribe_single_bassline(title, directories, scales, track_dicts, M,
                                    quantization_scheme, filter_unk, epsilon, pYIN_threshold,
                                    plot, date)             
                                    
    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


def prepare(directories_path):

    date = time.strftime("%Y-%m-%d_%H-%M-%S")

    directories = get_directories(directories_path)

    init_folders(directories['transcription'])
    init_exceptions_log(directories['transcription'], date)
            
    scales, track_dicts, track_titles = read_metadata(directories['transcription'])

    return directories, scales, track_dicts, track_titles, date


def plot_waveform_and_note_spectrogram(transcriber):

    center=True
    n_fft = 4096*8

    spectrogram_frame_factor = 8
    win_length = int((transcriber.beat_length/spectrogram_frame_factor)*transcriber.fs) 
    hop_length = int(win_length/4) 

    bassline_spectrogram = extract_dB_spectrogram(transcriber.bassline, n_fft, win_length, hop_length, center=center)

    waveform_and_note_spectrogram(transcriber.title, transcriber.directories,
                                transcriber.bassline, bassline_spectrogram,
                                transcriber.fs, hop_length,
                                transcriber.notes, transcriber.unk_notes,
                                show=False, save=True)

    close("all")
    clf()


if __name__ == '__main__':

    main()