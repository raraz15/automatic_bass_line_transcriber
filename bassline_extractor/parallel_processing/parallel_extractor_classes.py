#!/usr/bin/env python
# coding: utf-8

import os, sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from librosa import load 

# High Level Audio Processing
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor # Beat Tracking
from madmom.processors import SequentialProcessor

from bassline_extractor.chorus_estimation import drop_detection, check_chorus_beat_grid
from utilities import export_function, batch_export_function

from .parallel_madmom import process_batch
from .batch_source_separator import BatchSourceSeparator


class BatchBasslineExtractor:
    
    def __init__(self, titles, directories, fs=44100, N_bars=4, separator=None,
                track_dicts=None, thread_workers='auto', process_workers='auto'):
        """
        Parameters:
        -----------
            titles (lst): title of the tracks
            directories (dict): the  directories sub-dict corresponding to extraction process.
            track_dicts (dict): dictionary containing all tracks' information
            fs (int): sampling rate
            N_bars (int, default=4): Number of bars of bassline to extract
            separator (default=None): demucs Source Separator
            thread_workers (int, default='auto'): max workers for the track loader and the source separator
                let the cpu decide or infer from the batch_size
            process_workers (int default='cpu'): max processes to create, give an integer of let the cpu decide 
        """
        
        assert thread_workers in ['auto', 'batch'], 'thread_workers must be decided by\
                                                            the cpu or inferred from the batch_size'
        assert process_workers in ['auto', 'batch'], 'process_workers must be decided by\
                                                             the cpu or inferred from the batch_size'

        if thread_workers == 'auto':
            thread_workers = None
        else:
            thread_workers = len(titles)

        if process_workers != 'auto':
            process_workers = len(titles)
        
        self.info = BatchInfo(titles, directories['extraction'], fs, N_bars, track_dicts) # Track information class
        
        self.track = BatchTracks(self.info, thread_workers) # Track holder class

        self.beat_detector = BatchBeatDetector(self.info, process_workers) # Beat Grid Former
        
        self.chorus_detector = BatchChorusDetector(self.info, thread_workers) # Chorus Detector

        self.source_separator = BatchSourceSeparator(self.info, separator, thread_workers) # Source Separator is configured


class BatchInfo:
    """
    Information holder class. Stores track information for processing at further stages.
    """
    
    def __init__(self, titles, sub_directories, fs, N_bars, track_dicts):
        """
        titles (list): track titles in the batch
        """
        
        self.titles = titles
        self.directories = sub_directories
        self.N_bars = N_bars # number of bars to consider a chorus     
        self.fs = fs

        self.beat_lengths = None
        if track_dicts is not None: # if BPM value is provided
            self.beat_lengths={title: 60/int(track_dicts[title]['BPM']) for title in titles}


class BatchTracks:
    """
    Track loader class. Loads and stores the tracks using Multithreading.
    """
          
    def __init__(self, info, max_workers=None):
        
        self.info = info
        self.max_workers=max_workers

    def load_tracks(self):
        """
        Loads a batch of tracks.

            Parameters:
            -----------
                titles (list): title strings

            Returns:
            --------
                track_array_dict ({title: ndarray}) tracks corresponding to the titles
        """

        def loader(title, clip_dir, fs=44100):
            """ Loads a single track """
            path = os.path.join(clip_dir, title+'.mp3')
            track, _ = load(path, sr=fs, mono=True)
            track_tuple = (title, track)
            return track_tuple

        print('\nLoading a batch of tracks...')

        track_array_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor: 
            for title in self.info.titles:
                try:
                    future = executor.submit(loader, title, self.info.directories['clip'], self.info.fs)
                    title, track = future.result()
                    track_array_dict[title] = track
                except FileNotFoundError:
                    print('Track not Found: {}\nMoving to the next track.\n'.format(title))
                except Exception as ex:
                    print(''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__)))

        print('Done. (Loading)\n')
        return track_array_dict


class BatchBeatDetector:
    """
    BeatDetector class. Detects, stores and exports beat positions from a given track.
    """
    
    def __init__(self, info, max_workers='auto'):

        self.info = info
        self.max_workers = max_workers
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
        self.beat_positions_dict = process_batch(self.processor, track_array_dict, self.max_workers)
        print('Done. (Beat Positions)')
        return self.beat_positions_dict

    def export_beat_positions(self):
        """ Exports the beat positions and deletes them from the BeatDetector"""
        batch_export_function(self.beat_positions_dict, self.info.directories['beat_grid']['beat_positions'])
        del self.beat_positions_dict

    def load_beat_positions(self, track_array_dict):
        beat_positions_dict = {}
        for title in track_array_dict.keys():
            positions_path = os.path.join(self.info.directories['beat_grid']['beat_positions'], title+'.npy')
            beat_positions_dict[title] = np.load(positions_path)
        return beat_positions_dict
   
   
class BatchChorusDetector:
    """
    ChorusDetector class. Detects and extracts the chorus section from a given track.
    """
    
    def __init__(self, info, max_workers=None):
    
        self.info = info
        self.max_workers=max_workers

    def estimate_choruses(self, track_array_dict, beat_positions_dict):
        print('Estimating the Chorus positions...')

        def estimate_single_chorus(track, beat_positions, fs, N_bars, epsilon=2):            
            drop_beat_idx, _ = drop_detection(track, beat_positions, fs, epsilon)
            chorus_beat_positions = beat_positions[drop_beat_idx : drop_beat_idx+(N_bars*4)+1]
            return chorus_beat_positions

        chorus_estimates_dict = {}
        with ThreadPoolExecutor(self.max_workers) as executor: 
            for title, track in track_array_dict.items():
                future = executor.submit(estimate_single_chorus, track, beat_positions_dict[title],
                                        self.info.fs, self.info.N_bars)
                chorus_estimates_dict[title] = future.result()

        self.chorus_estimates_dict = chorus_estimates_dict
        print('Done. (Chorus)\n')

        if self.info.beat_lengths is not None: # Analyze beat positions if BPM is provided
            self.analyze_chorus_beats()

    def analyze_chorus_beats(self):
        for title, chorus_beat_positions in self.chorus_estimates_dict.items():
            deviation_indices = check_chorus_beat_grid(chorus_beat_positions, self.info.beat_lengths[title])
            if deviation_indices.size > 0:
                export_function(deviation_indices, self.info.directories['chorus']['chorus_beat_analysis'], title)
                print('Deviations in the chorus beat grid for:\n{}\n'.format(title))

    def export_chorus_beat_positions(self):
        batch_export_function(self.chorus_estimates_dict, self.info.directories['chorus']['chorus_beat_positions'])
    
    def extract_choruses(self, track_array_dict):
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

    def export_choruses(self):
        """Exports the choruses and deletes them from the BatchChorusDetector"""
        batch_export_function(self.chorus_dict, self.info.directories['chorus']['chorus_array'])
        del self.chorus_dict
