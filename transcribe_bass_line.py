import os
import argparse

from ablt.utilities import read_track_dicts
from ablt.bass_line_transcriber import transcribe_single_bass_line

DEFAULT_TRACK_DICTS_PATH = "data/metadata/track_dicts.json"
DEFAULT_BASS_LINE_DIR = "data/outputs"

M = [1,2,4,8]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-b', '--bassline-dir', type=str, help="Directory containing (an) / (all the) extracted bassline(s).", default=DEFAULT_BASS_LINE_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=DEFAULT_TRACK_DICTS_PATH)
    #parser.add_argument('-m', '--downsampling-rate', type=int, help='Downsampling rate to the F0 estimation.', default=1)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-f', '--frame-factor', type=int, help="Number of chorus bars to extract.", default=8)
    args = parser.parse_args()

    bassline_dir = args.bassline_dir
    #M = args.downsampling_rate
    N_bars = args.n_bars
    frame_factor = args.frame_factor

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.split(os.path.dirname(bassline_dir))[-1] == "outputs": # if a single file is specified

        title = os.path.basename(bassline_dir)
        track_dict = track_dicts[title] 

        bassline_path = os.path.join(bassline_dir, 'bass_line', title+'.npy')

        print('Lol')
        print(bassline_path)

        transcribe_single_bass_line(bassline_path, track_dict['BPM'], track_dict['Key'],
                                    M=M, N_bars=N_bars, frame_factor=frame_factor)

    else:
        track_titles = os.listdir(bassline_dir)
        for title in track_titles:

            track_dict = track_dicts[title]
            
            bassline_path = os.path.join(bassline_dir, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, track_dict['BPM'], track_dict['Key'],
                                        M=M, N_bars=N_bars, frame_factor=frame_factor)            
