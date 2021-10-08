# bassline_transcription

This repository contains an automatic bassline transcriber system that was designed for our Senior Design Project ELEC491 at Koç University, Istanbul / Turkey.

It estimates the beat positions using madmom, detects a drop by our custom algorithm then takes this drop as a chorus section and extracts the bassline using demucs_extra.

The isolated bassline in the beat grid then transcribed using pYIN, confidence filtered. Finally it is adaptively quantized by our custom algorithm and converted to a midi file

where middle C is taken as C4.

**How to Use:**

    1) Put your audio clips to data/audio_clips directory

    2) Create a track_dicts.json file

        This file must hold the BPM, key informatin of the tracks.
        An example can be found in data/metadata/track_dicts.json

    3) Extract the Bassline and Transcribe It

        python transcribe_bass_line.py 

        Check the arguments for detailed explanation.

    4.1) Bassline Extraction Only:

        You can specify an audio file or a folder containing multiple audio files to:

        python extract_bass_line.py 

        Check the arguments for detailed explanation.

    4.2) Transcription from Extracted Basslines

        You can specify an ouput folder or a directory containing multiple folders to:

        python transcribe_bass_line.py