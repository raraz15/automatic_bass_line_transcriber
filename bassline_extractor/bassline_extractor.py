#!/usr/bin/env python
# coding: utf-8

import os, sys, time

from tqdm import tqdm

from demucs.pretrained import load_pretrained

from utilities import prepare, init_folders, exception_logger
from .extractor_classes import BasslineExtractor, SimpleExtractor


def extract_single_bassline(title, directories, track_dicts, date, separator=None, fs=44100, N_bars=4):
    """
    Creates a Bassline_Extractor object for a track using the metadata provided. Extracts and Exports the Bassline.
    """

    try:

        init_folders(directories['extraction'])

        extractor = BasslineExtractor(title, directories, track_dicts, separator, fs, N_bars)

        # Estimate the Beat Positions and Export
        beat_positions = extractor.beat_detector.estimate_beat_positions(extractor.track.track)
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
        sys.exit()
        pass
    except KeyError as key_ex:
        print('Key Error on: {}'.format(title))
        exception_logger(directories['extraction'], key_ex, date, title, 'KeyError')
    except FileNotFoundError as file_ex:
        print('FileNotFoundError on: {}'.format(title))
        exception_logger(directories['extraction'], file_ex, date, title, 'FileNotFoundError')
    except RuntimeError as runtime_ex:
        print('RuntimeError on: {}'.format(title))
        exception_logger(directories['extraction'], runtime_ex, date, title, 'RuntimeError')
    except Exception as ex:     
        print("There was an unexpected error on: {}".format(title))
        exception_logger(directories['extraction'], ex, date, title, 'unexpected') 
  

def main(directories_path, track_dicts_name, idx=0):

    directories, _, track_dicts, track_titles, date = prepare(directories_path, track_dicts_name)

    separator = load_pretrained('demucs_extra')

    start_time = time.time()
    for title in tqdm(track_titles[idx:]):

        print('\n'+title)
        extract_single_bassline(title, directories, track_dicts, date, separator, fs=44100)

        with open('Completed_{}_{}.txt'.format(date, track_dicts_name.split('.json')[0]), 'a') as outfile:
            outfile.write(title+'\n')

    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


def separate_from_chorus(directories_path):

    directories, scales, track_dicts, track_titles, date = prepare(directories_path)

    separator = load_pretrained('demucs_extra')

    start_time = time.time()
    for title in tqdm(track_titles):

        print('\n'+title)

        extractor = SimpleExtractor(title, directories, track_dicts, separator)
        
        extractor.extract_and_export_bassline()
        
    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


if __name__ == '__main__':

    main()