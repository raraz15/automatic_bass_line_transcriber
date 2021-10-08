#!/usr/bin/env python
# coding: utf-8

import os
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

from representation import NN_output_to_midi_sequence
from bassline_transcriber.transcription import midi_sequence_to_midi_array


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

        note = int(midi_array[i, 1]) # convert to midi

        if middle_c == 'C3': 
            note += 12 

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


# TODO: remove or carry??
def NN_output_to_MIDI_file(representation, title, output_dir, M, 
                            BPM=125, N_qb=8, middle_c='C3', tpb=960*16,
                            min_note=28, silence_code=0, sustain_code=100, velocity=120):

    os.makedirs(output_dir, exist_ok=True)

    midi_sequence = NN_output_to_midi_sequence(representation, min_note, silence_code, sustain_code)
    midi_array = midi_sequence_to_midi_array(midi_sequence, M, N_qb=N_qb, silence_code=silence_code, velocity=velocity)
    create_MIDI_file(midi_array, BPM, title, output_dir, middle_c=middle_c, tpb=tpb)