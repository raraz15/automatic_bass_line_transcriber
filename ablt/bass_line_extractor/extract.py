#!/usr/bin/env python
# coding: utf-8

import os
import sys

from .extractor_class import BassLineExtractor
from ..utilities import exception_logger
from ..directories import OUTPUT_DIR


# TODO: track.track to track.audio ??
def extract_single_bass_line(path, N_bars=4, separator=None, BPM=0):
    """
    Creates a Bass line_Extractor object for a track using the metadata provided. Extracts and Exports the Bass line.
    """

    try:
        
        print('\n'+path)
        title = os.path.splitext(os.path.basename(path))[0]

        # Directory to log exceptions
        exception_dir = os.path.join(OUTPUT_DIR, "{}/exceptions/extraction".format(title))

        # Create the extractor
        extractor = BassLineExtractor(path, N_bars=N_bars, separator=separator, BPM=BPM)

        # Estimate the Beat Positions and Export
        beat_positions = extractor.beat_detector.estimate_beat_positions(extractor.track.track)
        extractor.beat_detector.export_beat_positions() 

        # Estimate the Chorus Position and Export
        chorus_beat_positions =extractor.chorus_detector.estimate_chorus_position(beat_positions)         
        extractor.chorus_detector.export_chorus_start_beat_idx()            
        extractor.chorus_detector.export_chorus_beat_positions()

        if BPM == 0:
            # Estimate the BPM and Export
            extractor.beat_detector.estimate_BPM(chorus_beat_positions)
            extractor.beat_detector.export_BPM()

        # Extract the Chorus and Export 
        chorus = extractor.chorus_detector.extract_chorus()
        extractor.chorus_detector.export_chorus()

        # Extract the Bass line from the Chorus 
        extractor.source_separator.separate_bass_line(chorus)   
        extractor.source_separator.process_bass_line()

        # Export the bass_line
        extractor.source_separator.export_bass_line()        

    except KeyboardInterrupt:
        sys.exit()
    except KeyError as key_ex:
        print('Key Error on: {}'.format(title))
        exception_logger(exception_dir, key_ex, title)
    except FileNotFoundError as file_ex:
        print('FileNotFoundError on: {}'.format(title))
        exception_logger(exception_dir, file_ex, title)
    except RuntimeError as runtime_ex:
        print('RuntimeError on: {}'.format(title))
        exception_logger(exception_dir, runtime_ex, title)
    except Exception as ex:     
        print("There was an unexpected error on: {}".format(title))
        exception_logger(exception_dir, ex, title)