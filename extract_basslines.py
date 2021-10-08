import argparse

from abt.utilities import get_directories, read_track_dicts
from abt.bassline_extractor import extract_all_basslines

DEFAULT_AUDO_DIR = "data/audio_clips"
DEFAULT_TRACK_DICTS_PATH = "data/metadata/track_dicts.json"

# Extracts the basslines of all wav and mp3 files in a directory using the metadata in the track_dicts.json file.

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=DEFAULT_AUDO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help="Path to the track_dicts.json file.", default=DEFAULT_TRACK_DICTS_PATH)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    args = parser.parse_args()

    track_dicts = read_track_dicts(args.track_dicts)
    directories = get_directories()

    extract_all_basslines(directories, args.audio_dir, track_dicts, args.n_bars)