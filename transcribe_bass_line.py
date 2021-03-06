import os
import argparse

import numpy as np

from ablt.utilities import read_track_dicts
from ablt.bass_line_transcriber import transcribe_single_bass_line

from ablt.directories import OUTPUT_DIR, TRACK_DICTS_PATH
from ablt.constants import HOP_RATIO, M


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-b', '--bassline-dir', type=str, help="Directory containing an / all extracted bassline(s).", default=OUTPUT_DIR)
    parser.add_argument('-t', '--track-dicts', action="store_true", help="Use a track_dicts.json file.")
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-m', '--downsampling-rate', type=int, help='Downsampling rate to the F0 estimation.', default=M)
    parser.add_argument('-f', '--hop-ratio', type=int, help="Number of F0 estimate samples that make up a beat.", default=HOP_RATIO)
    args = parser.parse_args()

    bassline_dir = args.bassline_dir
    M = args.downsampling_rate
    N_bars = args.n_bars
    hop_ratio = args.hop_ratio

    # Check if BPM values are provided
    if args.track_dicts:
        track_dicts = read_track_dicts(TRACK_DICTS_PATH)
    else:
        track_dicts = None

    if os.path.split(os.path.dirname(bassline_dir))[-1] == "outputs": # if a single file is specified
        
        title = os.path.splitext(os.path.basename(bassline_dir))[0]
        if track_dicts is None:
            BPM_path = os.path.join(OUTPUT_DIR, title, 'beat_grid', 'BPM.npy')
            BPM = np.load(BPM_path)
        else:
            BPM = track_dicts[title]['BPM']         

        bassline_path = os.path.join(bassline_dir, 'bass_line', title+'.npy')
        transcribe_single_bass_line(bassline_path, BPM=BPM, M=M,
                                    N_bars=N_bars, hop_ratio=hop_ratio)

    else:
        track_titles = os.listdir(bassline_dir)
        for title in track_titles:

            if track_dicts is None:
                BPM_path = os.path.join(OUTPUT_DIR, title, 'beat_grid', 'BPM.npy')
                BPM = np.load(BPM_path)
            else:
                BPM = track_dicts[title]['BPM']            
            
            bassline_path = os.path.join(bassline_dir, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, BPM=BPM, M=M,
                                        N_bars=N_bars, hop_ratio=hop_ratio)                                                    