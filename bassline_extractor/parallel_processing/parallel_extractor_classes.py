#!/usr/bin/env python
# coding: utf-8

import os, sys
import traceback, warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch import tensor
from librosa import load 
from librosa.util import normalize

# High Level Audio Processing
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor # Beat Tracking
from madmom.processors import SequentialProcessor

from demucs.utils import apply_model # Source Separation
from demucs.pretrained import load_pretrained

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from chorus_estimation import drop_detection, check_chorus_beat_grid
from signal_processing import lp_and_normalize
from utilities import export_function
from .parallel_madmom import process_batch

warnings.filterwarnings('ignore') # ignore librosa .mp3 warnings


class BatchBasslineExtractor:
    
    def __init__(self, title, directories, fs=44100, N_bars=4, separator=None):
        """
        Parameters:
        -----------
            title (str): title of the track
            directories (dict): the sub-dict corresponding to extraction process.
            track_dicts (dict): dictionary containing all tracks' information
            fs (int): sampling rate
            N_bars (int, default=4): Number of bars of bassline to extract
            separator (default=None): demucs Source Separator
        """
        
        self.info = BatchInfo(title, directories['extraction'], fs, N_bars) # Track information class
        
        self.track = BatchTracks(self.info) # Track holder class

        self.beat_detector = BatchBeatDetector(self.info) # Beat Grid Former
        
        self.chorus_detector = BatchChorusDetector(self.info) # Chorus Detector

        #self.source_separator = BatchSourceSeparator(self.info, separator) # Source Separator is configured


class BatchInfo:
    """
    Information holder class. Stores track information for processing at further stages.
    """
    
    def __init__(self, titles, directories, fs, N_bars, track_dicts=None):
        """
        titles (list): track titles in the batch
        """
        
        self.titles = titles
        self.track_dicts = track_dicts
        self.directories = directories
        self.N_bars = N_bars # number of bars to consider a chorus     
        self.fs = fs

        if track_dicts is not None: # if BPM value is provided
            self.beat_lengths={title: 60/int(track_dicts[title]['BPM']) for title in titles}


class BatchTracks:
    """
    Track loader class. Loads and stores the tracks using Multithreading.
    """
          
    def __init__(self, info):
        
        print('Loading a batch of tracks...')
        self.info = info
        self.track_array_dict = self.batch_track_loader(self.info.titles)

    # TODO: EXCEPTION HANDLING!!!!!!!!!!!!!!!!
    def batch_track_loader(self, titles):
        """
        Loads a batch of tracks.
            Parameters:
            -----------
                titles (list): title strings

            Returns:
            --------
                track_array_dict
        """

        def loader_with_try(title, clip_dir, fs=44100):
            path = os.path.join(clip_dir, title+'.mp3')
            try:
                track, _ = load(path, sr=fs, mono=True)
                track_tuple = (title, track)
            except:
                track_tuple = (None, None)       
            return  track_tuple

        track_array_dict = {}
        with ThreadPoolExecutor(max_workers=None) as executor: 
            for title in titles:
                future = executor.submit(loader_with_try, title, self.info.directories['clip'], self.info.fs)
                title, track = future.result()
                if title is not None:
                    track_array_dict[title] = track
        return track_array_dict


class BatchBeatDetector:
    """
    BeatDetector class. Detects, stores and exports beat positions from a given track.
    """
    
    def __init__(self, info):

        self.info = info
        self.processor = SequentialProcessor([RNNBeatProcessor(), BeatTrackingProcessor(fps=100)])
          
    def estimate_beat_positions(self, track_array_dict):
        """
        Estimates the beat positions, Fs must be 44100!!!.

            Parameters:
            -----------
                track_array_dict (dict): {title: ndarray} track dict                  
            Returns:
            --------
                beat_positions_dict (dict): {title: ndarray} beat positions dict
        """

        print('Estimating the beat positions...')
        self.beat_positions_dict = process_batch(self.processor, track_array_dict)
        return self.beat_positions_dict

    def export_beat_positions(self):
        batch_export_function(self.beat_positions_dict, self.info.directories['beat_grid']['beat_positions'])
   

class BatchChorusDetector:
    """
    ChorusDetector class. Detects and extracts the chorus section from a given track.
    """
    
    def __init__(self, info):
    
        self.info = info

    def estimate_choruses(self, track_array_dict, beat_positions_dict):
        print('Estimating the Chorus positions...')

        def estimate_single_chorus(track, beat_positions, fs, N_bars, epsilon=2):            
            drop_beat_idx, _ = drop_detection(track, beat_positions, fs, epsilon)
            chorus_beat_positions = beat_positions[drop_beat_idx : drop_beat_idx+(N_bars*4)+1]
            return chorus_beat_positions

        chorus_estimates_dict = {}
        with ThreadPoolExecutor(max_workers=None) as executor: 

            for title, track in track_array_dict.items():
                future = executor.submit(estimate_single_chorus, track, beat_positions_dict[title], self.info.fs, self.info.N_bars)
                chorus_estimates_dict[title] = future.result()

        self.chorus_estimates_dict = chorus_estimates_dict

        if self.info.track_dicts is not None: # Analyze beat positions if BPM provided
            self.analyze_chorus_beats()

    def analyze_chorus_beats(self, beat_factor=32):
        for title, chorus_beat_positions in self.chorus_estimates_dict.items():
            if check_chorus_beat_grid(chorus_beat_positions, self.info.beat_lengths[title], beat_factor).size > 0:
                export_function(chorus_beat_positions, self.info.directories['chorus']['chorus_beat_analysis'], self.info.title)
                print('Deviations in the chorus beat grid for {}'.format(title))

    def export_chorus_beat_positions(self):
        batch_export_function(self.chorus_estimates_dict, self.info.directories['chorus']['chorus_beat_positions'])
    
    def extract_chorus(self, track_array_dict):
        """
        Views the chorus from the loaded track given crorresponding beat positions in time.
        """
        chorus_dict = {}
        for title, chorus_beat_positions in self.chorus_estimates_dict.items():

            start_time, end_time = chorus_beat_positions[0], chorus_beat_positions[-1]
            start_idx, end_idx = int(start_time*self.info.fs), int(end_time*self.info.fs)

            chorus_dict[title] = track_array_dict[title][start_idx:end_idx+1]

        self.chorus_dict = chorus_dict
        return chorus_dict

    def export_chorus(self):
        batch_export_function(self.chorus_dict, self.info.directories['chorus']['chorus_array'])


class BatchSourceSeparator:
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


def batch_export_function(batch_dict, path):
    for title, array in batch_dict.items():
        export_function(array, path, title)

