#!/usr/bin/env python
# coding: utf-8

import os, sys
import warnings

import numpy as np
from torch import tensor
from librosa import load 
from librosa.util import normalize

# High Level Audio Processing
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor # Beat Tracking
from madmom.processors import SequentialProcessor

#from spleeter.separator import Separator # Source Separation
from demucs.utils import apply_model
from demucs.pretrained import load_pretrained

from .chorus_estimation import drop_detection, check_chorus_beat_grid
from signal_processing import lp_and_normalize
from utilities import export_function

warnings.filterwarnings('ignore') # ignore librosa .mp3 warnings


class BasslineExtractor:
    
    def __init__(self, title, directories, track_dicts, separator=None, fs=44100, N_bars=4):
        """
        Parameters:
        -----------
            title (str): title of the track
            directories (dict): the sub-dict corresponding to extraction process.
            track_dicts (dict): dictionary containing all tracks' information
            separator (default=None): demucs Source Separator
            fs (int): sampling rate
            N_bars (int, default=4): Number of bars of bassline to extract
        """
        
        self.info = Info(title, directories['extraction'], track_dicts, fs, N_bars) # Track information class
        
        self.track = Track(self.info) # Track holder class

        self.beat_detector = BeatDetector(self.info) # Beat Grid Former
        
        self.chorus_detector = ChorusDetector(self.info, self.track) # Chorus Detector

        self.source_separator = SourceSeparator(self.info, separator) # Source Separator is configured


class Info:
    """
    Information holder class. Stores track information for processing at further stages.
    """
    
    def __init__(self, title, directories, track_dicts, fs, N_bars):
        
        self.title = title
        self.track_dict = track_dicts[title]

        self.directories = directories
        self.path = os.path.join(directories['clip'], title+'.mp3')

        self.BPM = int(self.track_dict['BPM'])
        self.beat_length = 60 / self.BPM # length of one beat in sec
        self.N_bars = N_bars # number of bars to consider a chorus
        self.chorus_length = N_bars * (4 * self.beat_length)
        
        self.fs = fs
                 
        
class Track:
    """
    Track loader class. Loads and stores the track.
    """
          
    def __init__(self, info):
        
        print('Loading the track.')
        self.track, self.fs = load(info.path, sr=info.fs, mono=True)
        self.info = info


class BeatDetector:
    """
    BeatDetector class. Detects, stores and exports beat positions from a given track.
    """
    
    def __init__(self, info):

        self.info = info
        self.processor = SequentialProcessor([RNNBeatProcessor(), BeatTrackingProcessor(fps=100)])
          
    def estimate_beat_positions(self, track):
        """
        Estimates the beat positions.

            Parameters:
            -----------
                track (ndarray): 1D numpy array of the track, must have Fs=44100!
        """

        print('Finding the beat positions.')
        self.beat_positions = self.processor(track)
        
        return self.beat_positions

    def export_beat_positions(self):
        export_function(self.beat_positions, self.info.directories['beat_grid']['beat_positions'], self.info.title)
   

class ChorusDetector:
    """
    ChorusDetector class. Detects and extracts the chorus section from a given track.
    """
    
    def __init__(self, info, track):
    
        self.info = info
        self.track = track.track
        self.fs = track.fs

    def estimate_chorus(self, beat_positions, epsilon=2):
        """
        Estimates the chorus using the given beat positions.

            Parameters:
            -----------
                beat_positions (ndarray): beat positions in time
                epsilon (int, default=2): adjusts the threshold parameter for drop picking.
        """
        print('Estimating the Chorus position.')
        drop_beat_idx, _ = drop_detection(self.track, beat_positions, self.fs, epsilon)

        self.chorus_start_beat_idx = drop_beat_idx

        self.chorus_beat_positions = beat_positions[drop_beat_idx : drop_beat_idx+(self.info.N_bars*4)+1]

        self.analyze_chorus_beats()
        

    def extract_chorus(self):
        """
        Views the chorus from the loaded track given crorresponding beat positions in time.
        """

        start_time, end_time = self.chorus_beat_positions[0], self.chorus_beat_positions[-1]
        start_idx, end_idx = int(start_time*self.fs), int(end_time*self.fs)
        self.chorus = self.track[start_idx:end_idx+1]

        return self.chorus

    def export_chorus(self):
        export_function(self.chorus, self.info.directories['chorus']['chorus_array'], self.info.title)

    #??
    def analyze_chorus_beats(self):

        if check_chorus_beat_grid(self.chorus_beat_positions, self.info.beat_length).size > 0:
            export_function(self.chorus_beat_positions, self.info.directories['chorus']['chorus_beat_analysis'], self.info.title)

    def export_chorus_start_beat_idx(self):
        export_function(self.chorus_start_beat_idx, self.info.directories['chorus']['chorus_start_beat_idx'], self.info.title)

    def export_chorus_beat_positions(self):
        export_function(self.chorus_beat_positions, self.info.directories['chorus']['chorus_beat_positions'], self.info.title)
    

class SourceSeparator:
    """
    SourceSeparator class. Separates the bassline from a given chorus array and processes it.
    """
    
    def __init__(self, info, separator=None):
        """
            Parameters:
            -----------
                info (Info): Info class instance of the track.
                separator (default=None): provide a Source separator or load demucs_extra pretrained. 
        """
        
        self.info = info
        if separator is None:
            separator = load_pretrained('demucs_extra')
        self.separator = separator

    def separate_bassline(self, chorus):
        """
        Separates the bassline from a given chorus array.

            Parameters:
            -----------
                chorus (ndarray): chorus array

        source_names = ["drums", "bass", "other", "vocals"]
        """
        
        print('Separating the Bassline.') 

        # done in demucs implementation
        wav = np.stack([chorus]*2, axis=0)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()
        wav = tensor(wav)

        sources = apply_model(self.separator,
                            wav,
                            shifts=0,
                            split=True,
                            overlap=0.25,
                            progress=False)

        sources = sources * ref.std() + ref.mean()

        self.separated_bassline = sources[1,:,:].numpy()       

    def process_bassline(self):
        """
        Converts the extracted bassline to mono, normalizes it, LP filters at B2 and normalizes again.
        """

        bassline_mono = np.mean(self.separated_bassline, axis=0) # convert to mono
        bassline_mono_normalized = normalize(bassline_mono) # normalize bassline 
        
        fc = 130 # freq of B2 in Hz 

        self.bassline = lp_and_normalize(bassline_mono_normalized, fc, self.info.fs)

    def export_bassline(self):
        export_function(self.bassline, self.info.directories['bassline'], self.info.title)


class ChorusHolder:
    
    def __init__(self, title, fs, directories):
        
        self.chorus =  np.load(directories['chorus']['chorus_array']+'/'+title+'.npy')
        self.fs = fs


class SimpleExtractor:

    def __init__(self, title, directories, track_dicts, separator, fs=44100, N_bars=4):

        info = Info(title, directories, track_dicts, fs, N_bars)

        chorus_holder = ChorusHolder(title, fs, directories)

        self.separator = SourceSeparator(info, chorus_holder, separator)

    def extract_and_export_bassline(self):

        self.separator.separate_bassline()

        self.separator.process_bassline()

        self.separator.export_bassline()    