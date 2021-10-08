#!/usr/bin/env python
# coding: utf-8

import os, sys, time, glob

from tqdm import tqdm

from demucs.pretrained import load_pretrained

from .extractor_classes import BasslineExtractor

from utilities import get_directories

# TODO: track.track to track.audio ??
# TODO: update exception logger to not get date? 

def extract_single_bassline(path, directories, BPM, separator=None, fs=44100, N_bars=4):
    """
    Creates a Bassline_Extractor object for a track using the metadata provided. Extracts and Exports the Bassline.
    """

    try:
        
        print('\n'+path)
        title = os.path.splitext(os.path.basename(path))[0]

        # Create the extractor
        extractor = BasslineExtractor(path, directories, BPM, separator, fs, N_bars)

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
        #exception_logger(directories['extraction'], key_ex, date, title, 'KeyError')
    except FileNotFoundError as file_ex:
        print('FileNotFoundError on: {}'.format(title))
        #exception_logger(directories['extraction'], file_ex, date, title, 'FileNotFoundError')
    except RuntimeError as runtime_ex:
        print('RuntimeError on: {}'.format(title))
        #exception_logger(directories['extraction'], runtime_ex, date, title, 'RuntimeError')
    except Exception as ex:     
        print("There was an unexpected error on: {}".format(title))
        #exception_logger(directories['extraction'], ex, date, title, 'unexpected') 
  

def extract_all_basslines(audio_clips_dir, track_dicts):

    start_time = time.time()

    directories = directories = get_directories()
    
    # Load the demucs once here for faster training
    separator = load_pretrained('demucs_extra')

    # Get the list of wav and mp3 paths
    audio_paths = glob.glob(audio_clips_dir+'/*.mp3') + glob.glob(audio_clips_dir+'/*.wav')

    for path in tqdm(audio_paths):

        title = os.path.splitext(os.path.basename(path))[0]

        track_dict = track_dicts[title]

        extract_single_bassline(path, directories, track_dict['BPM'], separator, fs=44100, N_bars=4) 

    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))