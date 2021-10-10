import os
import argparse

from ablt.utilities import read_track_dicts
from ablt.bass_line_extractor import extract_single_bass_line, extract_all_bass_lines

from ablt.directories import TRACK_DICTS_PATH, AUDO_DIR

# Extracts the basslines of all wav and mp3 files in a directory using the metadata in the track_dicts.json file.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use the Bass Line Extractor on single or multiple audio files.")
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=AUDO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help="Path to the track_dicts.json file.", default=TRACK_DICTS_PATH)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.isfile(audio_dir): # if a single file is specified

        title = os.path.splitext(os.path.basename(audio_dir))[0]
        track_dict = track_dicts[title]        

        extract_single_bass_line(audio_dir, track_dict['BPM'], separator=None, N_bars=N_bars) 

    else: # if a folder of audio files is specified

        extract_all_bass_lines(audio_dir, track_dicts, N_bars=N_bars)