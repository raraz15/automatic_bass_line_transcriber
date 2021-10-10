import os
import argparse

from ablt.utilities import read_track_dicts

from ablt.bass_line_extractor import extract_single_bass_line
from ablt.bass_line_transcriber import transcribe_single_bass_line

from ablt.directories import OUTPUT_DIR, TRACK_DICTS_PATH, AUDIO_DIR
from ablt.constants import HOP_RATIO, M


# TODO: integrate parallel processing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=AUDIO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=TRACK_DICTS_PATH)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-f', '--hop-ratio', type=int, help="Number of F0 samples that makes up a beat.", default=HOP_RATIO)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars
    hop_ratio = args.hop_ratio

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.isfile(audio_dir): # if a single file is specified
        
        title = os.path.splitext(os.path.basename(audio_dir))[0]
        track_dict = track_dicts[title]        

        extract_single_bass_line(audio_dir, N_bars=N_bars, separator=None, BPM=track_dict['BPM'])
        
        bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
        transcribe_single_bass_line(bassline_path, BPM=track_dict['BPM'],
                                    M=M, N_bars=N_bars, hop_ratio=hop_ratio)

    else: # if a folder of audio files is specified

        audio_names= os.listdir(audio_dir)
        
        for audio_name in audio_names:

            title = os.path.splitext(audio_name)[0]
            track_dict = track_dicts[title]

            audio_path = os.path.join(audio_dir, audio_name)
            extract_single_bass_line(audio_path, N_bars=N_bars, separator=None, BPM=track_dict['BPM'])
            
            bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, BPM=track_dict['BPM'],
                                        M=M, N_bars=N_bars, hop_ratio=hop_ratio)