import argparse

from ablt.bass_line_extractor import main_batch


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Batch Bassline Extraction Parameters.')
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json')
    #parser.add_argument('-l', '--last-track', type=str, help='name of the last track ', default='')
    parser.add_argument('-b', '--batch-size', type=int, help='batch size for parallel processing', default=6)
    parser.add_argument('-w', '--thread-workers', type=str, help='auto or batch', default='auto')
    parser.add_argument('-v', '--process-workers', type=str, help='auto or batch', default='auto')
    args = parser.parse_args()

    track_dicts_name = args.track_dicts
    batch_size = args.batch_size
    last_track = args.last_track
    thread_workers = args.thread_workers
    process_workers = args.process_workers

    print('\n'+track_dicts_name+'\n')

    main_batch(DIRECTORIES_JSON_PATH, track_dicts_name, batch_size=batch_size, 
            thread_workers=thread_workers, process_workers=process_workers)