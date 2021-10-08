import argparse

from utilities import read_track_dicts
from bassline_extractor import extract_all_basslines

# Extracts the basslines of all wav and mp3 files in a directory using the metadata in the track_dicts.json file.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.")
    parser.add_argument('-t', '--track-dicts', type=str, help="Path to the track_dicts.json file.")
    args = parser.parse_args()

    track_dicts = read_track_dicts(args.track_dicts)

    extract_all_basslines(args.audio_dir, track_dicts)