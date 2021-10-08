#!/usr/bin/env python
# coding: utf-8

import os, sys, time, glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore') 

from matplotlib.pyplot import close, clf # required for preventing memory leakge while plotting

from .transcriber_class import BasslineTranscriber
from signal_processing import extract_dB_spectrogram
from plotting import waveform_and_note_spectrogram
from utilities import get_directories, exception_logger


def transcribe_single_bassline(path, directories, BPM, key, M=1, N_bars=4, frame_factor=8,
                                fs=44100, quantization_scheme='adaptive',
                                filter_unk=False, epsilon=4, pYIN_threshold=0.05, plot=False):

    try:

        print('\n'+path)
        title = os.path.splitext(os.path.basename(path))[0]

        bassline_transcriber = BasslineTranscriber(path, directories, BPM, key, M, N_bars, frame_factor, fs)

        # Pitch Track Extraction
        bassline_transcriber.extract_pitch_track(pYIN_threshold)
        bassline_transcriber.quantize_pitch_track(filter_unk, epsilon, quantization_scheme)

        # MIDI reconstruction
        bassline_transcriber.create_bassline_midi_file()

        # Exporting
        bassline_transcriber.export_F0_estimate()
        bassline_transcriber.export_pitch_track()
        bassline_transcriber.export_quantized_pitch_track()

        # Plotting
        if plot:
            bassline_transcriber.extract_notes()
            export_waveform_and_note_spectrogram(bassline_transcriber)

    except KeyboardInterrupt:
        sys.exit()
        pass
    except UnboundLocalError as u_ex:
        print("\nUnboundLocalError on: {}".format(title))
        exception_logger(directories['transcription'], u_ex, title)
    except IndexError as i_ex: 
        print("\nIndexError on: {}".format(title))
        exception_logger(directories['transcription'], i_ex, title)
    except FileNotFoundError as file_ex:
        print("\nFileNotFoundError on: {}".format(title))
        exception_logger(directories['transcription'], file_ex, title)
    except Exception as ex:     
        print("\nThere was an unexpected error on: {}".format(title))
        exception_logger(directories['transcription'], ex, title)


def transcribe_all_basslines(track_dicts, M=1, N_bars=4, frame_factor=8, fs=44100,
                            quantization_scheme='adaptive', filter_unk=False,
                            epsilon=4, pYIN_threshold=0.05, plot=False):

    start_time = time.time()            

    directories = get_directories()

    # Get the list of all bassline arrays
    bassline_paths = glob.glob(directories['extraction']['bassline']+'/*.npy')

    for path in tqdm(bassline_paths):

        title = os.path.splitext(os.path.basename(path))[0]

        track_dict = track_dicts[title]
        
        transcribe_single_bassline(path, directories, track_dict['BPM'], track_dict['Key'], M, N_bars,
                                    frame_factor, fs, quantization_scheme, filter_unk, epsilon, pYIN_threshold, plot)             
                                    
    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


def export_waveform_and_note_spectrogram(transcriber):

    center=True
    n_fft = 4096*8

    spectrogram_frame_factor = 8
    win_length = int((transcriber.beat_length/spectrogram_frame_factor)*transcriber.fs) 
    hop_length = int(win_length/4) 

    bassline_spectrogram = extract_dB_spectrogram(transcriber.bassline, n_fft, win_length, hop_length, center=center)

    waveform_and_note_spectrogram(transcriber.title, transcriber.complete_directories,
                                transcriber.bassline, bassline_spectrogram,
                                transcriber.fs, hop_length,
                                transcriber.notes, transcriber.unk_notes,
                                show=False, save=True)

    close("all")
    clf()