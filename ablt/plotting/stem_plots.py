#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt

from .building_blocks import beat_plotting, save_function
from ..utilities import get_quarter_beat_positions, sample_and_hold


def chorus_bassline_stem(title, beat_positions, chorus, bassline, N_beats, fs):
    
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

    fig.suptitle(title)

    #plt.savefig('x.jpg')
    plt.show()


def plot_beat_grid(beat_positions, ax, min_amp, max_amp):

    bar_positions, beat_positions, qb_positions = beat_plotting(beat_positions)

    ax.vlines(bar_positions, min_amp, max_amp, alpha=0.7, color='g',linestyle='dashed', linewidths=2)
    ax.vlines(beat_positions, min_amp*0.9, max_amp*0.9, alpha=0.7, color='r',linestyle='dashed', linewidths=2)
    ax.vlines(qb_positions, min_amp*0.7, max_amp*0.7, alpha=0.7, color='k',linestyle='dashed', linewidths=2)
    

def F0_related_stem(track_title, beat_positions, F0_estimate, pitch_track, quantized_pitch_track, midi_sequence, M,
                    plot_title='', plot_dir='', ):

    midi_sequence = sample_and_hold(midi_sequence, M)
    
    fig, ax = plt.subplots(nrows=4, figsize=(20,16), constrained_layout=True)
    fig.suptitle(track_title+'\n\nTranscription Steps', fontsize=20)

    t = F0_estimate[0] # t is shared between all F0 variants

    ax[0].stem(t, F0_estimate[1])
    plot_beat_grid(beat_positions, ax[0], 0, F0_estimate[1].max())
    ax[0].set_ylabel('Hz', fontsize=14)
    ax[0].set_title('F0 Estimate', fontsize=16)

    ax[1].stem(t, pitch_track[1])
    plot_beat_grid(beat_positions, ax[1], 0, pitch_track[1].max())
    ax[1].set_ylabel('Hz', fontsize=14)
    ax[1].set_title('Pitch Track', fontsize=16)

    ax[2].stem(t, quantized_pitch_track[1])
    plot_beat_grid(beat_positions, ax[2], 0, quantized_pitch_track[1].max())
    ax[2].set_ylabel('Hz', fontsize=14)
    ax[2].set_title('Quantized Pitch Track', fontsize=16)

    ax[3].stem(t, midi_sequence)
    plot_beat_grid(beat_positions, ax[3], 0, max(midi_sequence))
    ax[3].set_ylabel('Midi Number', fontsize=14)
    ax[3].set_title('Midi Number Sequence(M={})'.format(M), fontsize=16)

    for x in ax:
        x.set_xlabel('Time(s)', fontsize=14)
        x.grid()
        x.set_xlim([-t[2], t[-1]+t[1]])

    if plot_dir:
        save_function(plot_dir, track_title, plot_title=plot_title, default_title="TranscriptionSteps")

    #plt.savefig('quantization_process.jpg')
    plt.show()