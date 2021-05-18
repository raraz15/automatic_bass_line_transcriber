#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

from demucs.pretrained import load_pretrained

from utilities import get_directories, init_folders, init_exceptions_log, read_metadata
from .extractor_classes import BasslineExtractor, SimpleExtractor

project_dir = '/mnt/d/projects/bassline_extraction'


def extract_single_bassline(title, directories, track_dicts, scales, date, separator=None, fs=22050, N_bars=4):
    """
    Creates a Bassline_Extractor object for a track using the metadata provided. Extracts and Exports the Bassline.
    """

    try:

        extractor = BasslineExtractor(title, directories, track_dicts, scales, separator, fs, N_bars)

        # Estimate the Beat Positions and Export
        beat_positions = extractor.beat_detector.estimate_beat_positions()
        extractor.beat_detector.export_beat_positions() 


        # Estimate the Chorus Position and Extract
        extractor.chorus_detector.estimate_chorus(beat_positions, epsilon=2)  
        extractor.chorus_detector.export_chorus_start_beat_idx()            
        extractor.chorus_detector.export_chorus_beat_positions()

        # Extract the Chorus and Export 
        chorus = extractor.chorus_detector.extract_chorus()
        extractor.chorus_detector.export_chorus()


        # Extract the Bassline from the Chorus 
        extractor.source_separator.separate_bassline(chorus)   
        extractor.source_separator.process_bassline()

        # Export the bassline
        extractor.source_separator.export_bassline()           

    except KeyboardInterrupt:
        import sys
        sys.exit()
        pass               
    except Exception as ex:     
        print("There was an error on: {}".format(title))
        exception_str = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        print(exception_str+'\n')
        with open(os.path.join(directories['exceptions'], '{}.txt'.format(date)), 'a') as outfile:
            outfile.write(title+'\n'+exception_str+'\n\n') 


def main(directories_path=project_dir, idx=0):

    directories, track_dicts, scales, track_titles, date = prepare(directories_path)

    separator = load_pretrained('demucs_extra')

    start_time = time.time()
    for title in tqdm(track_titles[idx:]):

        print('\n'+title)
        
        extract_single_bassline(title, directories, track_dicts, scales, date, separator, fs=22050)

    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))



def separate_from_chorus(directories_path=project_dir):

    directories, track_dicts, scales, track_titles, date = prepare(directories_path)

    separator = load_pretrained('demucs_extra')

    start_time = time.time()
    for title in tqdm(track_titles):

        print('\n'+title)

        extractor = SimpleExtractor(title, directories, track_dicts, scales, separator)
        
        extractor.extract_and_export_bassline()
        
    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


def prepare(directories_path):

    date = time.strftime("%Y-%m-%d_%H-%M-%S")

    directories = get_directories(directories_path)['extraction']

    init_folders(directories)
    init_exceptions_log(directories, date)
            
    scales, track_dicts, track_titles = read_metadata(directories)

    return directories, track_dicts, scales, track_titles, date


if __name__ == '__main__':

    main()