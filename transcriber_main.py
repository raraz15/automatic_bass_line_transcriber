import argparse

from bassline_transcriber import main

parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
parser.add_argument('-d', '--directories', type=str, help='Path to directories.json', default='data/directories.json')
parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json')
#parser.add_argument('-l', '--last-track', type=str, help='name of the last track ', default='')

args = parser.parse_args()

directories_path=args.directories
track_dicts_name = args.track_dicts

print('\n'+track_dicts_name+'\n')
main(directories_path, track_dicts_name, M=[1,2,4,8])