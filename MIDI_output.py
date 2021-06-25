#!/usr/bin/env python
# coding: utf-8

import os
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo


def create_MIDI_file(midi_array, BPM, title, output_dir, middle_c='C3', tpb=960*16):
              
    outfile = MidiFile(ticks_per_beat=tpb)
    track = MidiTrack()

    outfile.tracks.append(track)

    track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(BPM)))
    track.append(MetaMessage('track_name', name=title))
    track.append(MetaMessage('instrument_name', name='bass'))
    
    delta0 = int(midi_array[0, 0]*tpb)
    for i in range(midi_array.shape[0]-1):

        note = int(midi_array[i, 1]) # convert to midi

        if middle_c == 'C4': 
            note -= 12 

        duration = int(midi_array[i, 3]*tpb)

        offset = midi_array[i, 0] + midi_array[i, 3]

        track.append(Message('note_on', note=note, velocity=100, time=delta0))
        track.append(Message('note_off', note=note, velocity=100, time=duration))

        # Delta time for midi
        delta0 = int((midi_array[i+1, 0]-offset)*tpb)

    duration = int(midi_array[i+1, 3]*tpb)
    track.append(Message('note_on', note=note, velocity=100, time=delta0))
    track.append(Message('note_off', note=note, velocity=100, time=duration))
    
    output_path = os.path.join(output_dir, '{}.mid'.format(title))
    outfile.save(output_path)