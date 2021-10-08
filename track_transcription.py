import os
import argparse

from ablt.utilities import read_track_dicts

from ablt.bass_line_extractor import extract_single_bass_line, extract_all_bass_lines
from ablt.bass_line_transcriber import transcribe_single_bass_line

DEFAULT_AUDO_DIR = "data/audio_clips"
DEFAULT_TRACK_DICTS_PATH = "data/metadata/track_dicts.json"
DEFAULT_OUTPUT_DIR = "data/outputs"

M = [1,2,4,8]

# TODO: parallel processing
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bassline Transcription Parameters.')
    parser.add_argument('-a', '--audio-dir', type=str, help="Directory containing all the audio files.", default=DEFAULT_AUDO_DIR)
    parser.add_argument('-t', '--track-dicts', type=str, help='Path to track_dicts.json', default=DEFAULT_TRACK_DICTS_PATH)
    parser.add_argument('-n', '--n-bars', type=int, help="Number of chorus bars to extract.", default=4)
    parser.add_argument('-f', '--frame-factor', type=int, help="Number of chorus bars to extract.", default=8)
    args = parser.parse_args()

    audio_dir = args.audio_dir
    N_bars = args.n_bars
    frame_factor = args.frame_factor

    track_dicts = read_track_dicts(args.track_dicts)

    if os.path.isfile(audio_dir): # if a single file is specified
        
        title = os.path.splitext(os.path.basename(audio_dir))[0]
        track_dict = track_dicts[title]        

        extract_single_bass_line(audio_dir, track_dict['BPM'], separator=None, N_bars=N_bars)
        
        bassline_path = os.path.join(DEFAULT_OUTPUT_DIR, title, 'bass_line', title+'.npy')
        transcribe_single_bass_line(bassline_path, track_dict['BPM'], track_dict['Key'],
                                    M=M, N_bars=N_bars, frame_factor=frame_factor)

    else: # if a folder of audio files is specified

        audio_names= os.listdir(audio_dir)
        
        for audio_name in audio_names:

            title = os.path.splitext(audio_name)[0]
            track_dict = track_dicts[title]

            audio_path = os.path.join(audio_dir, audio_name)
            extract_single_bass_line(audio_path, track_dict['BPM'], separator=None, N_bars=N_bars)
            
            bassline_path = os.path.join(DEFAULT_OUTPUT_DIR, title, 'bass_line', title+'.npy')
            transcribe_single_bass_line(bassline_path, track_dict['BPM'], track_dict['Key'],
                                        M=M, N_bars=N_bars, frame_factor=frame_factor)