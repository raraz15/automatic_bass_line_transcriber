#!/usr/bin/env python
# coding: utf-8

import os
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
from ..signal_processing import lp_and_normalize
from ..utilities import export_function

from ..constants import FS, CUTOFF_FREQ
from ..directories import OUTPUT_DIR

warnings.filterwarnings('ignore') # ignore librosa .mp3 warnings


# TODO: wav writing the bassline and the chorus
class BassLineExtractor:
    
    def __init__(self, path, N_bars=4, separator=None, BPM=0):
        """
        Parameters:
        -----------
        
            path (str): path of the track
            N_bars (int, default=4): Number of bars of bass line to extract
            separator (default=None): demucs Source Separator
            BPM (float, default=0):  track BPM (optional)
            
        """
        
        self.info = Info(path, BPM, FS, N_bars) # Track information class
        
        self.track = Track(self.info) # Track holder class

        self.beat_detector = BeatDetector(self.info) # Beat Grid Former
        
        self.chorus_detector = ChorusDetector(self.info, self.track) # Chorus Detector

        self.source_separator = SourceSeparator(self.info, separator) # Source Separator is configured

class Info:
    """
    Information holder class. Stores track information for processing at further stages.
    """
    
    def __init__(self, path, BPM, fs, N_bars):
        
        self.path = path # track path
        self.title = os.path.splitext(os.path.basename(path))[0]
         
        BPM = float(BPM)
        if BPM != 0.:
            self.BPM = BPM
            self.beat_length = 60 / self.BPM # length of one beat in sec
        else:
            self.BPM = None

        self.N_bars = N_bars # number of bars to consider a chorus
        
        self.fs = fs

        # Form the Export directories
        self.output_dir = os.path.join(OUTPUT_DIR, self.title)
        self.beatgrid_dir = os.path.join(self.output_dir, 'beat_grid')
        self.chorus_dir = os.path.join(self.output_dir, 'chorus')
        self.bass_line_dir = os.path.join(self.output_dir, 'bass_line')

        for d in [self.output_dir, self.beatgrid_dir, self.chorus_dir, self.bass_line_dir]:
            os.makedirs(d, exist_ok=True)

        #CHORUS_AUDIO_DIR = os.path.join(self.chorus_dir, 'audio') # LATER
        self.chorus_array_dir = os.path.join(self.chorus_dir, 'array')
        self.chorus_beat_analysis_dir = os.path.join(self.chorus_dir, 'beat_analysis')
        self.chorus_start_beat_idx_dir = os.path.join(self.chorus_dir, 'start_beat_idx')
        self.chorus_beat_positions_dir = os.path.join(self.chorus_dir, 'beat_positions')
               
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
                track (ndarray): 1D numpy array of the track, must have Fs=44100 for beat detection.

            Returns:
            --------
                beat_positions (ndarray): beat positions array
        """

        print('Finding the beat positions.')
        self.beat_positions = self.processor(track)
        return self.beat_positions

    def estimate_BPM(self, beat_positions):
        """
        Estimates the BPM using given beat positions. Recommended to use chorus beat positions.

            Parameters:
            -----------
                beat_positions (ndarray): beat positions array

            Returns:
            --------
                BPM (float): estimated BPM value
        """
        mean_beat_length = np.mean(np.diff(beat_positions))
        self.BPM = np.round(60/mean_beat_length)
        print('{} BPM is estimated.'.format(self.BPM))
        return self.BPM        

    def export_beat_positions(self):
        export_function(self.beat_positions, self.info.beatgrid_dir, self.info.title)

    def export_BPM(self):
        export_function(self.BPM, self.info.beatgrid_dir, 'BPM')

class ChorusDetector:
    """
    ChorusDetector class. Detects and extracts the chorus section from a given track.
    """
    
    def __init__(self, info, track):
    
        self.info = info
        self.track = track.track
        self.fs = track.fs

    # TODO: chorus epsilon parameter is different
    def estimate_chorus_position(self, beat_positions, epsilon=2):
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

        # return the first beat of the next bar too
        self.chorus_beat_positions = beat_positions[drop_beat_idx : drop_beat_idx+(self.info.N_bars*4)+1] 

        if self.info.BPM is not None: # Analyze estimated beat positions if BPM is provided
            self.analyze_chorus_beats()

        return self.chorus_beat_positions
        
    def extract_chorus(self):
        """
        Views the chorus from the loaded track given crorresponding beat positions in time.
        """

        start_time, end_time = self.chorus_beat_positions[0], self.chorus_beat_positions[-1]
        start_idx, end_idx = int(start_time*self.fs), int(end_time*self.fs)
        self.chorus = self.track[start_idx:end_idx+1]
        return self.chorus

    def export_chorus(self):
        export_function(self.chorus, self.info.chorus_array_dir, self.info.title)

    def analyze_chorus_beats(self):
        assert self.info.BPM is not None, 'You must provide a BPM value for analyzing the extracted beat grid!'
        if check_chorus_beat_grid(self.chorus_beat_positions, self.info.beat_length).size > 0:
            export_function(self.chorus_beat_positions, self.info.chorus_beat_analysis_dir, self.info.title)

    def export_chorus_start_beat_idx(self):
        export_function(self.chorus_start_beat_idx, self.info.chorus_start_beat_idx_dir, self.info.title)

    def export_chorus_beat_positions(self):
        export_function(self.chorus_beat_positions, self.info.chorus_beat_positions_dir, self.info.title)
    
class SourceSeparator:
    """
    SourceSeparator class. Separates the bass line from a given chorus array and processes it.
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

    def separate_bass_line(self, chorus):
        """
        Separates the bass line from a given chorus array.

            Parameters:
            -----------
                chorus (ndarray): chorus array

        source_names = ["drums", "bass", "other", "vocals"]
        """
        
        print('Separating the Bass Line.') 

        # For demucs implementation
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

        self.separated_bass_line = sources[1,:,:].numpy()       

    def process_bass_line(self):
        """
        Converts the extracted bass line to mono, normalizes it, LP filters at B2 and normalizes again.
        """

        bass_line_mono = np.mean(self.separated_bass_line, axis=0) # convert to mono
        bass_line_mono_normalized = normalize(bass_line_mono) # normalize bass line  

        self.bass_line = lp_and_normalize(bass_line_mono_normalized, CUTOFF_FREQ, self.info.fs)

    def export_bass_line(self):
        print("Exporting the Bass Line.")
        export_function(self.bass_line, self.info.bass_line_dir, self.info.title) 