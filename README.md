# Automatic Bass Line Transcription for Electronic Music

This repository contains an automatic bassline transcriber system that was designed for our Spring 21' Senior Design Project ELEC491 at Koç University, Istanbul / Turkey.

It estimates the beat positions using madmom, detects a drop by our custom algorithm then takes this drop as a chorus section and extracts the bassline using demucs_extra.

The isolated bassline in the beat grid then transcribed using pYIN, confidence filtered. Finally it is adaptively quantized by our custom algorithm and converted to a midi file

where middle C is taken as C4.


**How to Use:**

    1) Importing Audio Files:

        The files will be resampled to 44100Hz. This sample rate is required for the beat detection model.

        You can either:
            A) Put your audio clips to data/audio_clips directory or
            B) Specify their paths to the scripts using --audio-dir [audio_path]

    2) Extract a Bassline and Transcribe It

        You can specify an audio file or a folder containing multiple audio files to:

        python transcribe_bass_line.py --audio-dir=[audio_dir]

        Check the arguments for detailed explanation.

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
