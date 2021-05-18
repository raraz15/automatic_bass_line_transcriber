#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict

from .quantization.pitch_quantization import quantize_frequency


def extract_note_dicts(F0_estimate, track_scale, epsilon=2):
    """
    Creates dictionaries of (time,pitch) mappings from an F0_estimate tupple.
    Used for plotting.
    """

    note_names, scale_frequencies, out_of_scale_note_names, out_of_scale_frequencies = track_scale
           
    notes_dict = {n: {'time': [], 'frequency': []} for n in ['-'] + note_names}
    unk_notes_dict = defaultdict(lambda: {'time': [], 'frequency': []})
    
    for idx, f in enumerate(F0_estimate[1]):

        t = F0_estimate[0][idx]
        
        if f: # if non-zero

            if f not in scale_frequencies:

                # look for close notes outside the scale
                f = quantize_frequency(f, out_of_scale_frequencies, epsilon)

                if f in out_of_scale_frequencies:

                    note_idx = out_of_scale_frequencies.index(f) # index of the corresponding note in the scale
                    note_name = out_of_scale_note_names[note_idx]

                    unk_notes_dict[note_name]['time'].append(t)
                    unk_notes_dict[note_name]['frequency'].append(f) 

                else:  # if the pitch is not in the diatonic scale, erase it

                    notes_dict['-']['time'].append(t)
                    notes_dict['-']['frequency'].append(0.0)    
                             
            else: # f in the scale

                note_idx = scale_frequencies.index(f) # index of the corresponding note in the scale
                note_name = note_names[note_idx]

                notes_dict[note_name]['time'].append(t)
                notes_dict[note_name]['frequency'].append(f)

        else: # f is zero

            notes_dict['-']['time'].append(t)
            notes_dict['-']['frequency'].append(f)            
                                  
    return notes_dict, unk_notes_dict