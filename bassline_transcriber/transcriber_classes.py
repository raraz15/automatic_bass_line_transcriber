#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore') 

import numpy as np

from utilities import (get_chorus_beat_positions, get_quarter_beat_positions, 
                      get_track_scale, export_function)
from .transcription import (pYIN_F0, adaptive_voiced_region_quantization,
                            uniform_voiced_region_quantization, midi_sequence_to_midi_array,
                            extract_note_dicts, frequency_to_midi_sequence)


class BasslineTranscriber():

    def __init__(self, title, directories, scales, track_dicts, M, fs=44100, N_bars=4, frame_factor=8, silence_code=0):

        self.title = title
        self.fs = fs
        self.directories = directories['transcription']
        self.complete_directories = directories # for plotting

        self.key, self.scale_type = track_dicts[title]['Key'].split(' ')
        self.track_scale = get_track_scale(title, track_dicts, scales)
        
        self.M = [M] if isinstance(M, int) else M # decimation rates
        self.frame_factor = frame_factor # F0 estimation frame size w.r.t a beat
        self.N_bars = N_bars
        self.BPM = float(track_dicts[title]['BPM'])        
        self.beat_length = 60/self.BPM

        self.quarter_beat_positions = get_quarter_beat_positions(get_chorus_beat_positions(title, directories))
        self.bassline = np.load(directories['extraction']['bassline']+'/'+title+'.npy')
        
        self.silence_code=silence_code


    def extract_pitch_track(self, pYIN_threshold=0.05):

        #Initial estimate | Confidence Filtered
        self.F0_estimate, self.pitch_track = pYIN_F0(self.bassline,
                                                    self.fs,
                                                    beat_length=self.beat_length,
                                                    N_bars=self.N_bars,
                                                    threshold=pYIN_threshold)

        
    def quantize_pitch_track(self, filter_unk, epsilon, quantization_scheme):

        assert quantization_scheme in ['adaptive', 'uniform'], 'Choose between adaptive and uniform quantization!'

        if quantization_scheme == 'adaptive':
            self.pitch_track_quantized = adaptive_voiced_region_quantization(self.pitch_track,
                                                                self.track_scale,
                                                                self.quarter_beat_positions,
                                                                filter_unk, 
                                                                length_threshold=self.frame_factor,
                                                                epsilon=epsilon)
        else:
            self.pitch_track_quantized = uniform_voiced_region_quantization(self.pitch_track, self.track_scale, epsilon)


    def create_bassline_midi_file(self):
        from MIDI_output import create_MIDI_file

        midi_sequence = frequency_to_midi_sequence(self.pitch_track_quantized[1], self.silence_code)
        for m in self.M:
            bassline_midi_array = midi_sequence_to_midi_array(midi_sequence, M=m, N_qb=self.frame_factor,
                                                                silence_code=self.silence_code)

            create_MIDI_file(bassline_midi_array, self.BPM, self.title, self.directories['midi']['midi_file'][str(m)])        


    def export_F0_estimate(self):
        export_function(self.F0_estimate, self.directories['bassline_transcription']['F0_estimate'], self.title)

    def export_pitch_track(self):
        export_function(self.pitch_track, self.directories['bassline_transcription']['pitch_track'], self.title)

    def export_quantized_pitch_track(self):
        export_function(self.pitch_track_quantized, self.directories['bassline_transcription']['quantized_pitch_track'], self.title)    


    def extract_notes(self):
        """ Finds the notes in and out the scale, mainly for plotting."""
        self.notes, self.unk_notes = extract_note_dicts(self.pitch_track_quantized, self.track_scale)