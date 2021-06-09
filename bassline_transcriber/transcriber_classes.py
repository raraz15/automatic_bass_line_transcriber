#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore') 

import numpy as np

from utilities import (get_chorus_beat_positions, get_quarter_beat_positions, 
                      get_track_scale, export_function)
from MIDI_output import create_MIDI_file
from .transcription import (pYIN_F0, adaptive_voiced_region_quantization,
                            uniform_voiced_region_quantization, 
                            extract_note_dicts, frequency_to_midi_sequence,
                            midi_sequence_to_midi_array,
                            encode_midi_sequence, downsample_midi_number_sequence)


class BasslineTranscriber():

    def __init__(self, title, directories, scales, track_dicts, M, fs=44100, N_bars=4, frame_factor=8, sustain_code=100, silence_code=0):

        self.title = title
        self.fs = fs
        self.directories = directories['transcription']
        self.complete_directories = directories # for plotting

        self.key, self.scale_type = track_dicts[title]['Key'].split(' ')
        self.track_scale = get_track_scale(title, track_dicts, scales)
        
        if isinstance(M, int):
            M = [M]
        self.M = M # decimation rates
        self.frame_factor = frame_factor 
        self.N_bars = N_bars
        self.BPM = float(track_dicts[title]['BPM'])        
        self.beat_length = 60/self.BPM

        self.quarter_beat_positions = get_quarter_beat_positions(get_chorus_beat_positions(title, directories))
        self.bassline = np.load(directories['extraction']['bassline']+'/'+title+'.npy')

        self.sustain_code=sustain_code
        self.silence_code=silence_code


    def extract_pitch_track(self, pYIN_threshold=0.05):

        frame_length = int((self.beat_length/self.frame_factor)*self.fs)

        #Initial estimate, Confidence Filtered
        self.F0_estimate, self.pitch_track = pYIN_F0(self.bassline,
                                                    self.fs,
                                                    frame_length,
                                                    threshold=pYIN_threshold)

    def quantize_pitch_track(self, filter_unk, epsilon, quantization_scheme):

        assert quantization_scheme in ['adaptive', 'uniform'], 'Choose between adaptive and uniform quantization!'
        if quantization_scheme == 'adaptive':
            self.quantize_pitch_track_adaptively(filter_unk, epsilon)
        else:
            self.quantize_pitch_track_uniformly(epsilon)


    def quantize_pitch_track_adaptively(self, filter_unk=False, epsilon=2):

        self.pitch_track_quantized = adaptive_voiced_region_quantization(self.pitch_track,
                                                                        self.track_scale,
                                                                        self.quarter_beat_positions,
                                                                        filter_unk, 
                                                                        length_threshold=self.frame_factor,
                                                                        epsilon=epsilon)

    def quantize_pitch_track_uniformly(self, epsilon=4):

        self.pitch_track_quantized = uniform_voiced_region_quantization(self.pitch_track, self.track_scale, epsilon)

    def create_midi_sequence(self):
        self.bassline_midi_sequence = frequency_to_midi_sequence(self.pitch_track_quantized[1], silence_code=self.silence_code)

    def create_midi_array(self):
        """Creates a single MIDI_array if M is an int or for each M creates a midi array."""

        bassline_midi_array = {}
        for m in self.M:
            bassline_midi_array[m] = midi_sequence_to_midi_array(self.bassline_midi_sequence, 
                                                m, N_qb=self.frame_factor,
                                                silence_code=self.silence_code, velocity=120)
        self.bassline_midi_array=bassline_midi_array                
        
    def create_symbolic_representation(self):
        representation = {}
        for m in self.M:
            midi_sequence =  downsample_midi_number_sequence(self.bassline_midi_sequence, m, self.frame_factor, self.N_bars)  
            representation[m] = encode_midi_sequence(midi_sequence, sustain_code=self.sustain_code, key=self.key)
        self.representation = representation


    def extract_notes(self):
        """ Finds the notes in and out the scale, mainly for plotting."""
        self.notes, self.unk_notes = extract_note_dicts(self.pitch_track_quantized, self.track_scale)


    def export_F0_estimate(self):
        export_function(self.F0_estimate, self.directories['bassline_transcription']['F0_estimate'], self.title)

    def export_pitch_track(self):
        export_function(self.pitch_track, self.directories['bassline_transcription']['pitch_track'], self.title)

    def export_quantized_pitch_track(self):
        export_function(self.pitch_track_quantized, self.directories['bassline_transcription']['quantized_pitch_track'], self.title)

    def export_midi_array(self):
        for m in self.M:
            export_function(self.bassline_midi_array[m], self.directories['midi']['midi_array'][str(m)], self.title)
        
    def create_midi_file(self):
        for m in self.M:
            create_MIDI_file(self.bassline_midi_array[m], self.BPM, self.title, self.directories['midi']['midi_file'][str(m)])

    def export_symbolic_representation(self):
        for m in self.M:
            export_function(self.representation[m], self.directories['symbolic_representation'][str(m)], self.title) 