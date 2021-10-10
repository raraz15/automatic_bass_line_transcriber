#!/usr/bin/env python
# coding: utf-8

import os, sys

from matplotlib.pyplot import close, clf # required for preventing memory leakge while plotting

from .transcriber_class import BassLineTranscriber
from ..signal_processing import extract_dB_spectrogram
from ..plotting import waveform_and_note_spectrogram
from ..utilities import exception_logger
from ..directories import OUTPUT_DIR
from ..constants import HOP_RATIO, M


def transcribe_single_bass_line(path, BPM, M=M, N_bars=4, hop_ratio=HOP_RATIO,
                                quantization_scheme='adaptive', epsilon=2,
                                pYIN_threshold=0.05, plot=False):
    """
        Parameters:
        -----------
            path (str): full path to the file including the extension
            BPM (float): BPM of the track
            M (int, default=1): Downsampling ratio
            N_bars (int, default=4): Number of chorus bars to transcribe
            hop_ratio (int): Number of F0 samples that will make up a beat
            quantization_scheme (str, default=adaptive): F0 quantization scheme
            epsilon (int): freq_bound = delta_scale/epsilon determines if quantization will happen.
            pYIN_threshold (float, default=0.05): Confidence level threshold for F0 estimation filtering.
            plot (bool, default=False): plot the bassline or not.

    """

    try:

        print('\n'+path)
        title = os.path.splitext(os.path.basename(path))[0]

        # Directory to log exceptions
        exception_dir = os.path.join(OUTPUT_DIR, "{}/exceptions/transciption".format(title))

        bass_line_transcriber = BassLineTranscriber(path, BPM, M=M, N_bars=N_bars, hop_ratio=hop_ratio)

        # Pitch Track Extraction
        bass_line_transcriber.extract_pitch_track(pYIN_threshold)
        bass_line_transcriber.quantize_pitch_track(epsilon, quantization_scheme)
        
        # Convert to MIDI pitches
        bass_line_transcriber.create_MIDI_sequence()

        # Exporting
        bass_line_transcriber.export_F0_estimate()
        bass_line_transcriber.export_pitch_track()
        bass_line_transcriber.export_quantized_pitch_track()

        # MIDI reconstruction
        bass_line_transcriber.create_bass_line_MIDI_file()        

        # Plotting
        if plot:
            bass_line_transcriber.extract_notes()
            export_waveform_and_note_spectrogram(bass_line_transcriber)

        print('Transcription complete.')

    except KeyboardInterrupt:
        sys.exit()
    except UnboundLocalError as u_ex:
        print("UnboundLocalError!")
        exception_logger(exception_dir, u_ex, title)
    except IndexError as i_ex: 
        print("IndexError!")
        exception_logger(exception_dir, i_ex, title)
    except FileNotFoundError as file_ex:
        print("FileNotFoundError!")
        exception_logger(exception_dir, file_ex, title)
    except Exception as ex:     
        print("There was an unexpected error!")
        exception_logger(exception_dir, ex, title)


def export_waveform_and_note_spectrogram(transcriber):

    center=True
    n_fft = 4096*8

    spectrogram_frame_factor = 8
    win_length = int((transcriber.beat_length/spectrogram_frame_factor)*transcriber.fs) 
    hop_length = int(win_length/4) 

    bass_line_spectrogram = extract_dB_spectrogram(transcriber.bass_line, n_fft, win_length, hop_length, center=center)

    waveform_and_note_spectrogram(transcriber.title, transcriber.complete_directories,
                                transcriber.bass_line, bass_line_spectrogram,
                                transcriber.fs, hop_length,
                                transcriber.notes, transcriber.unk_notes,
                                show=False, save=True)

    close("all")
    clf()