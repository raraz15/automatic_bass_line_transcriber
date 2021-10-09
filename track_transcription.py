import os
import argparse

from ablt.utilities import read_track_dicts

from ablt.bass_line_extractor import extract_single_bass_line
from ablt.bass_line_transcriber import transcribe_single_bass_line

from ablt.constants import OUTPUT_DIR, METADATA_DIR, AUDO_DIR
TRACK_DICTS_PATH = os.path.join(METADATA_DIR, "track_dicts.json")

M = [1,2,4,8]

# TODO: parallel processing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=AUDO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=TRACK_DICTS_PATH)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-f', '--hop-factor', type=int, help="Number of chorus bars to extract.", default=64)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars
    hop_factor = args.hop_factor

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.isfile(audio_dir): # if a single file is specified
        
        title = os.path.splitext(os.path.basename(audio_dir))[0]
        track_dict = track_dicts[title]        

        extract_single_bass_line(audio_dir, BPM=track_dict['BPM'], separator=None, N_bars=N_bars)
        
        bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
        transcribe_single_bass_line(bassline_path, BPM=track_dict['BPM'], key=track_dict['Key'],
                                    M=M, N_bars=N_bars, hop_factor=hop_factor)

    else: # if a folder of audio files is specified

        audio_names= os.listdir(audio_dir)
        
        for audio_name in audio_names:

            title = os.path.splitext(audio_name)[0]
            track_dict = track_dicts[title]

            audio_path = os.path.join(audio_dir, audio_name)
            extract_single_bass_line(audio_path, BPM=track_dict['BPM'], separator=None, N_bars=N_bars)
            
            bassline_path = os.path.join(OUTPUT_DIR, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, BPM=track_dict['BPM'], key=track_dict['Key'],
                                        M=M, N_bars=N_bars, hop_factor=hop_factor)