import argparse

from bassline_transcriber import main

DIRECTORIES_JSON_PATH = 'data/directories.json'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json')
    args = parser.parse_args()

    track_dicts_name = args.track_dicts

    print('\n'+track_dicts_name+'\n')
    main(DIRECTORIES_JSON_PATH, track_dicts_name, M=[1,2,4,8])