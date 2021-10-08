import os
import argparse

from ablt.utilities import read_track_dicts, get_directories

from ablt.bass_line_extractor import extract_single_bassline
from ablt.bass_line_transcriber import transcribe_all_basslines, transcribe_single_bassline

DEFAULT_AUDO_DIR = "data/audio_clips"
DEFAULT_TRACK_DICTS_PATH = "data/metadata/track_dicts.json"

# TODO: M
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=DEFAULT_AUDO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=DEFAULT_TRACK_DICTS_PATH)
    args = parser.parse_args()

    audio_dir = args.audio_dir

    track_dicts = read_track_dicts(args.track_dicts)
    directories = get_directories()

    if os.path.isfile(audio_dir): # if a single file is specified

        title = os.path.splitext(os.path.basename(audio_dir))[0]

        track_dict = track_dicts[title]        

        extract_single_bassline(audio_dir, directories, track_dict['BPM'], separator=None, fs=44100, N_bars=4) 
        transcribe_single_bassline()

    else: # if a folder of audio files is specified
        extract_all_basslines(directories, audio_dir, track_dicts)
        transcribe_all_basslines(track_dicts, M=[1,2,4,8])