#!/usr/bin/env python
# coding: utf-8


import numpy as np
from matplotlib import pyplot as plt
from utilities import get_quarter_beat_positions, sample_and_hold

def chorus_bassline_stem(title, chorus, bassline, beat_positions, N_beats, fs):
    
    quarter_beat_positions = get_quarter_beat_positions(beat_positions)
    
    plot_quarter_beats = (quarter_beat_positions - quarter_beat_positions[0])*fs
    plot_beats = (beat_positions- beat_positions[0])*fs

    N = int(plot_beats[N_beats])

    fig, ax = plt.subplots(nrows=2, figsize=(20, 8), constrained_layout=True)

    ax[0].plot(chorus[:N])
    ax[0].set_title('Chorus')
    ax[0].vlines(plot_quarter_beats,-1,1, colors='k')
    ax[0].vlines(plot_beats,-1,1, colors='r', linewidth=2)
    ax[0].set_xlim([-150, N+150])
    ax[0].set_xlabel('Samples')

    ax[1].plot(bassline[len(bassline) - len(chorus):][:N])
    ax[1].set_title('Bassline')
    ax[1].vlines(plot_quarter_beats,-1,1, colors='k')
    ax[1].vlines(plot_beats,-1,1, colors='r', linewidth=2)
    ax[1].set_xlim([-150, N+150])
    ax[1].set_xlabel('Samples')

    plt.suptitle(title)

    #plt.savefig('x.jpg')
    plt.show()

def F0_related_stem(title, F0_estimate, pitch_track, quantized_pitch_track, midi_sequence, M):
    
    #N_quarterbeats = N_bars*32
    
    midi_sequence = sample_and_hold(midi_sequence, M)
    
    fig, ax = plt.subplots(nrows=4, figsize=(20,16), constrained_layout=True)

    ax[0].stem(F0_estimate[1])
    ax[0].vlines(np.arange(0, 129, 8),0,60, colors='k')
    ax[0].vlines(np.arange(0, 129, 32),0,60, colors='r', linewidth=2)
    ax[0].set_xlim([-1, 132])
    ax[0].set_ylabel('Hz', fontsize=14)
    ax[0].set_title('F0 Estimate', fontsize=15)

    ax[1].stem(pitch_track[1])
    ax[1].vlines(np.arange(0, 129, 8),0,60, colors='k')
    ax[1].vlines(np.arange(0, 129, 32),0,60, colors='r', linewidth=2)
    ax[1].set_xlim([-1, 132])
    ax[1].set_ylabel('Hz', fontsize=14)
    ax[1].set_title('Pitch Track', fontsize=15)

    ax[2].stem(quantized_pitch_track[1])
    ax[2].vlines(np.arange(0, 129, 8),0,60, colors='k')
    ax[2].vlines(np.arange(0, 129, 32),0,60, colors='r', linewidth=2)
    ax[2].set_xlim([-1, 132])
    ax[2].set_ylabel('Hz', fontsize=14)
    ax[2].set_title('Quantized Pitch Track', fontsize=15)

    ax[3].stem(midi_sequence)
    ax[3].set_xlim([-1, len(midi_sequence)+1])
    ax[3].vlines(np.arange(0, 129, 8),0,60, colors='k')
    ax[3].vlines(np.arange(0, 129, 32),0,60, colors='r', linewidth=2)
    ax[3].set_xlim([-1, 132])
    ax[3].set_ylabel('Midi Number', fontsize=14)
    ax[3].set_title('Midi Number Sequence(M={})'.format(M), fontsize=15)

    plt.suptitle(title, fontsize=17)

    plt.savefig('quantization_process.jpg')
    plt.show()