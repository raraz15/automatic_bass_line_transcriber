import os
import argparse

from ablt.utilities import read_track_dicts
from ablt.bassline_transcriber import transcribe_single_bassline, transcribe_all_basslines

DEFAULT_TRACK_DICTS_PATH = "data/metadata/track_dicts.json"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-b', '--bassline-dir', type=str, help="Directory containing (an) / (all the) extracted bassline(s).", default=)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=DEFAULT_TRACK_DICTS_PATH)
    args = parser.parse_args()

    bassline_dir = args.bassline_dir

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.isfile(bassline_dir): # if a single file is specified
        transcribe_single_bassline
    else:

        transcribe_all_basslines(track_dicts, M=[1,2,4,8])