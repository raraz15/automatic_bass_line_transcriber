import argparse

from bassline_extractor import main_batch
from utilities import find_track_index

parser = argparse.ArgumentParser(description='Batch Bassline Extraction Parameters.')
parser.add_argument('-d', '--directories', type=str, help='Path to directories.json', default='data/directories.json')
parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json')
parser.add_argument('-l', '--last-track', type=str, help='name of the last track ', default='')
parser.add_argument('-b', '--batch-size', type=int, help='batch size for parallel processing', default=6)
parser.add_argument('-w', '--thread-workers', type=str, help='auto or batch', default='auto')
parser.add_argument('-v', '--process-workers', type=str, help='auto or batch', default='auto')

args = parser.parse_args()

directories_path=args.directories
track_dicts_name = args.track_dicts
batch_size = args.batch_size
last_track = args.last_track
thread_workers = args.thread_workers
process_workers = args.process_workers


if last_track:
    idx = find_track_index(last_track, directories_path, track_dicts_name)+1
else:
    idx = 0

print('\n'+track_dicts_name+'\n')

main_batch(directories_path, track_dicts_name, idx=idx, batch_size=batch_size, 
        thread_workers=thread_workers, process_workers=process_workers)