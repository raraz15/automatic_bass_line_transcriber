#!/usr/bin/env python
# coding: utf-8

import os

from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo


def create_MIDI_file(midi_array, BPM, title, output_dir, middle_c='C4', tpb=960*16):
              
    outfile = MidiFile(ticks_per_beat=tpb)
    track = MidiTrack()
    outfile.tracks.append(track)

    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(BPM)))
    track.append(MetaMessage('track_name', name=title))
    track.append(MetaMessage('instrument_name', name='bass'))
    
    delta0 = int(midi_array[0, 0]*tpb)
    for i in range(midi_array.shape[0]-1):

        pitch = int(midi_array[i, 1]) # convert to midi pitch

        if middle_c == 'C3': 
            pitch += 12 

        duration = int(midi_array[i, 3]*tpb)

        offset = midi_array[i, 0] + midi_array[i, 3]

        track.append(Message('note_on', note=pitch, velocity=100, time=delta0))
        track.append(Message('note_off', note=pitch, velocity=100, time=duration))

        # Delta time for midi
        delta0 = int((midi_array[i+1, 0]-offset)*tpb)

    # Put the last note
    duration = int(midi_array[i+1, 3]*tpb)
    pitch = int(midi_array[i+1, 1])
    track.append(Message('note_on', note=pitch, velocity=100, time=delta0))
    track.append(Message('note_off', note=pitch, velocity=100, time=duration))

    track.append(MetaMessage('end_of_track'))

    os.makedirs(output_dir, exist_ok=True)  
    output_path = os.path.join(output_dir, '{}.mid'.format(title))
    outfile.save(output_path)