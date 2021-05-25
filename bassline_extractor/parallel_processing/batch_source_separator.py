#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch import tensor

from librosa.util import normalize
from signal_processing import lp_and_normalize

from demucs.utils import apply_model # Source Separation
from demucs.pretrained import load_pretrained
from utilities import batch_export_function

class BatchSourceSeparator:
    """
    SourceSeparator class. Separates the bassline from a given chorus array and processes it.
    """
    
    def __init__(self, info, separator=None, max_workers=None):
        """
            Parameters:
            -----------
                info (Info): Info class instance of the track.
                separator (default=None): provide a Source separator or load demucs_extra pretrained.
                max_workers (int, default=None): number of workers for multithreading, give None for
                                                        letting the computer decide.
        """
        
        self.info = info
        if separator is None:
            separator = load_pretrained('demucs_extra')
        self.separator = separator
        self.max_workers=max_workers
    
    def separate_basslines(self, chorus_dict):
        print('Separating the Basslines...')
        
        bassline_dict = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor: 

            for title, chorus in chorus_dict.items():
                future = executor.submit(separate_single_bassline, chorus, self.separator, self.info.fs)
                bassline_dict[title] = future.result()
        
        self.bassline_dict = bassline_dict
                    
    def export_basslines(self):
        """ Exports and deletes the basslines from the BatchSourceSeparator"""
        batch_export_function(self.bassline_dict, self.info.directories['bassline'])
        del self.bassline_dict


def separate_single_bassline(chorus, separator, fs):
    """
    Separates the bassline from a given chorus array.

        Parameters:
        -----------
            chorus (ndarray): chorus array

    source_names = ["drums", "bass", "other", "vocals"]
    """

    chorus, mean, std = preprocess_chorus(chorus)

    sources = apply_model(separator, chorus,
                        shifts=0, split=True,
                        overlap=0.25, progress=False)

    sources = sources*std + mean

    separated_bassline = sources[1,:,:].numpy()

    processed_bassline = process_bassline(separated_bassline, fs)
    
    return processed_bassline
      
        
def preprocess_chorus(chorus, audio_channels=2):

    if audio_channels == 2:
        chorus = np.stack([chorus]*2, axis=0)            
    ref = chorus.mean(0)
    mean, std = ref.mean(), ref.std()
    chorus = (chorus - mean) / std

    return tensor(chorus), mean, std


def process_bassline(separated_bassline, fs):
    """
    Converts the extracted bassline to mono, normalizes it, LP filters at B2 and normalizes again.
    """

    bassline_mono = np.mean(separated_bassline, axis=0) # convert to mono
    bassline_mono_normalized = normalize(bassline_mono) # normalize bassline 

    fc = 130 # freq of B2 in Hz 
    processed_bassline = lp_and_normalize(bassline_mono_normalized, fc, fs)
    
    return processed_bassline