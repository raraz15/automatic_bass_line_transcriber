import os
import argparse
import tqdm

from ablt.utilities import read_track_dicts, load_source_separation_model
from ablt.bass_line_extractor import extract_single_bass_line

from ablt.directories import AUDIO_DIR, TRACK_DICTS_PATH

# Extracts the basslines of all wav and mp3 files in a directory using the metadata in the track_dicts.json file.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Use the Bass Line Extractor on single or multiple audio files.")
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=AUDIO_DIR)
    parser.add_argument('-t', '--track-dicts', action="store_true", help="Use a track_dicts.json file.")
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars

    # Check if BPM value is provided
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

    else: # if a directory of audio files is specified

        # Load the demucs once here for faster training
        separator = load_source_separation_model       

        # Get the list of all wav and mp3 paths
        track_titles = os.listdir(audio_dir)
        for title in tqdm.tqdm(track_titles):

            audio_path = os.path.join(audio_dir, title)

            if track_dicts is None:
                BPM = 0
            else:
                BPM = track_dicts[title]['BPM']

            extract_single_bass_line(audio_path, N_bars=N_bars, separator=separator, BPM=BPM) 