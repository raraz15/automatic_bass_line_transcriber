#!/usr/bin/env python
# coding: utf-8


def NN_output_to_midi_sequence(representation, min_note=28, silence_code=0, sustain_code=100):
    _representation = representation.copy()
    midi_sequence = expand_consecutive_symbols(_representation, min_note=min_note, silence_code=silence_code, sustain_code=sustain_code)
    if sustain_code is not None:
        midi_sequence = replace_sustain(midi_sequence, sustain_code)
    return midi_sequence


def expand_consecutive_symbols(representation, min_note=28, silence_code=0, sustain_code=100):
    """Converts NN output (consecutive integers) to midi code  0,28,29,..., 51,100  for example"""

    code = representation.copy()
    code[code!=silence_code] += min_note-1
    if sustain_code is not None:
        max_code = code.max()
        code[code==max_code] = sustain_code # max will be the sustain
    return code


def replace_sustain(codes, sustain_code=100):

    arr = codes.copy()
    if len(arr.shape) == 2:
        for r in arr:
            for idx, el in enumerate(r[1:]):
                if el == sustain_code:
                    r[idx + 1] = r[idx]
    else:
        for idx, el in enumerate(arr[1:]):
            if el == sustain_code:
                arr[idx + 1] =arr[idx]        

    return arr