import os
import argparse

import numpy as np

from ablt.utilities import read_track_dicts

from ablt.bass_line_extractor import extract_single_bass_line
from ablt.bass_line_transcriber import transcribe_single_bass_line

from ablt.directories import OUTPUT_DIR, TRACK_DICTS_PATH, AUDIO_DIR
from ablt.constants import HOP_RATIO, M


# TODO: integrate parallel processing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=AUDIO_DIR)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-f', '--hop-ratio', type=int, help="Number of F0 samples that makes up a beat.", default=HOP_RATIO)
    parser.add_argument('-t', '--track-dicts', action="store_true", help="Use a track_dicts.json file.")
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars
    hop_ratio = args.hop_ratio

    # Load BPM values if provided
    if args.track_dicts:
        track_dicts = read_track_dicts(TRACK_DICTS_PATH)           
    else:
        track_dicts = None  

    if os.path.isfile(audio_dir): # if a single file is specified

        if track_dicts is None:
            BPM = 0
        else:
            title = os.path.splitext(os.path.basename(audio_dir))[0]
            BPM = track_dicts[title]['BPM']
        
        extract_single_bass_line(audio_dir, N_bars=N_bars, separator=None, BPM=BPM)

        # Update with the estimated BPM
        if track_dicts is None:
            BPM_path = os.path.join(OUTPUT_DIR, title, 'beat_grid', 'BPM.npy')
            BPM = np.load(BPM_path)        
        
        bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
        transcribe_single_bass_line(bassline_path, BPM=BPM, M=M,
                                    N_bars=N_bars, hop_ratio=hop_ratio)

    else: # if a folder of audio files is specified

        audio_names = os.listdir(audio_dir)
        for title_ext in audio_names:

            title = os.path.splitext(title_ext)[0]

            if track_dicts is None:
                BPM = 0
            else:
                BPM = track_dicts[title]['BPM']

            audio_path = os.path.join(audio_dir, title_ext)
            extract_single_bass_line(audio_path, N_bars=N_bars, separator=None, BPM=BPM)

            # Update with the estimated BPM
            if track_dicts is None:
                BPM_path = os.path.join(OUTPUT_DIR, title, 'beat_grid', 'BPM.npy')
                BPM = np.load(BPM_path)       
            
            bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, BPM=BPM, M=M, 
                                        N_bars=N_bars, hop_ratio=hop_ratio)