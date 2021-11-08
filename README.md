# Automatic Bass Line Transcription for Electronic Music

This repository contains an automatic bass line transcriber system that was designed for the Spring 21' Senior Design Project ELEC491 at Koç University, Istanbul / Turkey.

It estimates the beat positions using madmom, detects a drop by our custom drop detection algorithm then takes this drop as a chorus section and extracts the bassline using demucs_extra.

The isolated chorus bassline, which is locked tightly to the beat grid, is then transcribed using pYIN and is confidence filtered.

Finally it is adaptively quantized by our custom quantization algorithm and converted to a midi file where middle C is taken as C4.

The project will be presented at the ISMIR 2021 conference as a Late Breaking Demo (LBD). The accepted paper and the poster are included in the documents folder.

**Installing Dependencies**

    1) Initialize an environment

    2) Activate

    3) Install dependencies using requirements.txt

**How to Use:**

    1) Importing Audio Files:

        The files will be resampled to 44100Hz. This sample rate is critical for the beat detection model.

        You can either:
            A) Put your audio clips to data/audio_clips directory or
            B) Specify their paths to the scripts using --audio-dir [audio_path]

    2) Extract a Bassline and Transcribe It

        You can specify an audio file or a folder containing multiple audio files to:

        python automatic_bass_line_transcription.py --audio-dir=[audio_dir]

        Check the arguments for detailed explanation.

        All the output files will be exported to data/outputs/track_title/

    3.1) Bassline Extraction Only:

        You can specify an audio file or a folder containing multiple audio files to:

        python extract_bass_line.py --audio-dir=[audio_dir]

        Check the arguments for detailed explanation.

    3.2) Transcription from Extracted Basslines

        You can specify an ouput folder or a directory containing multiple folders to:

        python transcribe_bass_line.py

        Check the arguments for detailed explanation.

    4) (Optional) Provide BPM annotations.

        You can provide the ABLT with known BPM information but the model is capable of estimating the BPM itself.
        
        If you choose to do so,
        
        Create a track_dicts.json file in the data/metadata folder.
        Which should be a dictionary of {track_title: {'BPM': BPM_value}}
        An example can be found in data/metadata/track_dicts.json

        To use this dictionary for the three scripts, use the --track-dicts flag.        
