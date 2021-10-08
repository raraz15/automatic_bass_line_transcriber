#!/usr/bin/env python
# coding: utf-8

import sys, time
import gc

import numpy as np
from tqdm import tqdm

from demucs.pretrained import load_pretrained

from .parallel_extractor_classes import BatchBasslineExtractor

from ...utilities import init_folders, exception_logger

DIRECTORIES_JSON_PATH = 'data/directories.json'

def extract_batch_basslines(titles, directories, date, fs=44100, N_bars=4, separator=None, track_dicts=None,
                            thread_workers='auto', process_workers='auto'):
    """
    Creates a Bassline_Extractor object for a batch of tracks using the metadata provided. Extracts and Exports the Bassline.
    """

    try:

        init_folders(directories['extraction'])

        extractor = BatchBasslineExtractor(titles, directories, fs, N_bars, separator, track_dicts,
                                            thread_workers, process_workers)

        # Return the loaded tracks
        track_array_dict = extractor.track.load_tracks()

        # Estimate the Beat Positions and Export
        #beat_positions_dict = extractor.beat_detector.load_beat_positions(track_array_dict)
        beat_positions_dict = extractor.beat_detector.estimate_beat_positions(track_array_dict)
        extractor.beat_detector.export_beat_positions() 


        # Estimate the Chorus Positions and Extract
        extractor.chorus_detector.estimate_choruses(track_array_dict, beat_positions_dict)                    
        extractor.chorus_detector.export_chorus_beat_positions()

        # Extract the Choruses and Export 
        chorus_dict = extractor.chorus_detector.extract_choruses(track_array_dict)
        extractor.chorus_detector.export_choruses()

        # Extract the Basslines from the Choruses
        extractor.source_separator.separate_basslines(chorus_dict)   

        # Export the basslines
        extractor.source_separator.export_basslines()

        del chorus_dict
        del track_array_dict
        del beat_positions_dict
        del extractor
        gc.collect()

    except KeyboardInterrupt:
        sys.exit()
        pass
    except KeyError as key_ex:
        print('\nKey Error inside batch. Check the exception log for more detail.')
        exception_logger(directories['extraction'], key_ex, date, '\n'.join(titles))
    except FileNotFoundError as file_ex:
        print('\nFileNotFoundError inside batch. Check the exception log for more detail.')
        exception_logger(directories['extraction'], file_ex, date, '\n'.join(titles))
    except RuntimeError as runtime_ex:
        print('\nRuntimeError inside batch. Check the exception log for more detail.')
        exception_logger(directories['extraction'], runtime_ex, date, '\n'.join(titles))
    except Exception as ex:     
        print("\nThere was an unexpected error inside batch. Check the exception log for more detail.")
        exception_logger(directories['extraction'], ex, date, '\n'.join(titles)) 


def main(track_dicts_name, batch_size=6, thread_workers='auto', process_workers='auto'):
    
    directories, _, track_dicts, track_titles, date = prepare(DIRECTORIES_JSON_PATH, track_dicts_name)

    separator = load_pretrained('demucs_extra') # load demucs once at the beginning

    N_batches = len(track_titles) // batch_size

    start_time = time.time()
    for batch_titles in tqdm(np.array_split(track_titles, N_batches)):

        extract_batch_basslines(batch_titles, directories, date, separator=separator, track_dicts=track_dicts,
                                thread_workers=thread_workers, process_workers=process_workers)

        with open('Completed_{}_{}.txt'.format(date, track_dicts_name.split('.json')[0]), 'a') as outfile:
            outfile.write('\n'.join(batch_titles)+'\n')

    print('Total Run:', time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time)))


if __name__ == '__main__':

    main()