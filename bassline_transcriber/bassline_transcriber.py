#!/usr/bin/env python
# coding: utf-8

import os, sys, time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore') 

from matplotlib.pyplot import close, clf # required for preventing memory leakge while plotting

from .transcriber_classes import BasslineTranscriber
from signal_processing import extract_dB_spectrogram
from plotting import waveform_and_note_spectrogram
from utilities import prepare, init_folders, exception_logger


def transcribe_single_bassline(title, directories, scales, track_dicts, M=1, quantization_scheme='adaptive',
                                filter_unk=False, epsilon=4, pYIN_threshold=0.05, plot=False, date=''):

    try:

        init_folders(directories['transcription'])

        print('\n'+title)

        bassline_transcriber=BasslineTranscriber(title, directories, scales, track_dicts, M)

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
        exception_logger(directories['transcription'], u_ex, date, title)
    except IndexError as i_ex: 
        print("\nIndexError on: {}".format(title))
        exception_logger(directories['transcription'], i_ex, date, title)
    except FileNotFoundError as file_ex:
        print("\nFileNotFoundError on: {}".format(title))
        exception_logger(directories['transcription'], file_ex, date, title)
    except Exception as ex:     
        print("\nThere was an unexpected error on: {}".format(title))
        exception_logger(directories['transcription'], ex, date, title)


def main(directories_path, track_dicts_name, M=1, quantization_scheme='adaptive', filter_unk=False,
            epsilon=4, pYIN_threshold=0.05, plot=False):

    directories, scales, track_dicts, track_titles, date = prepare(directories_path, track_dicts_name)

    start_time = time.time()
    for title in tqdm(track_titles):

        transcribe_single_bassline(title, directories, scales, track_dicts, M,
                                    quantization_scheme, filter_unk, epsilon, pYIN_threshold,
                                    plot, date)             
                                    
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


if __name__ == '__main__':

    main()