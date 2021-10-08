import argparse

from utilities import read_track_dicts
from bassline_transcriber import transcribe_all_basslines

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json')
    args = parser.parse_args()

    track_dicts = read_track_dicts(args.track_dicts)

    transcribe_all_basslines(track_dicts, M=[1,2,4,8])