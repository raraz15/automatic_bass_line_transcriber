#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from matplotlib import pyplot as plt

import librosa.display

from ..utilities import get_bar_positions, get_quarter_beat_positions

colors = ['0.5','tab:orange','tab:olive','moccasin','khaki','steelblue','b','g','r','c','m','y','k','c','w']
unk_colors = ['purple','hotpink','lime','firebrick','salmon','darkred','mistyrose']


def beat_plotting(beat_positions):
    """
    Makes the beat-grid plottable.
    """
    
    beat_positions -= beat_positions[0]
    bar_positions = get_bar_positions(beat_positions)
    beat_positions_plotting = [val for idx,val in enumerate(beat_positions) if idx%4]
    quarter_beat_positions = [val for idx,val in enumerate(get_quarter_beat_positions(beat_positions)) if idx%4]  
        
    return bar_positions, beat_positions_plotting, quarter_beat_positions

#def plot_beat_grid(beat_positions, ax):
#
#    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(beat_positions)
#
#    ax.vlines(beat_positions_plotting, -0.9, 0.9, alpha=0.8, color='r',linestyle='dashed', linewidths=3)
#    ax.vlines(quarter_beat_positions, -0.7, 0.7, alpha=0.8, color='k',linestyle='dashed', linewidths=3)
#    ax.vlines(bar_positions, -1.1, 1.1, alpha=0.8, color='g',linestyle='dashed', linewidths=3)

def form_beat_grid_waveform(beat_positions, audio_array, fs, ax):
    """
    Plots the bar, beat and quarter beats on a given waveform plt.ax
    """

    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(beat_positions)

    librosa.display.waveplot(audio_array, sr=fs, ax=ax)
    
    ax.vlines(bar_positions, -1.1, 1.1, alpha=0.8, color='g',linestyle='dashed', linewidths=3, label='Bar Positions')
    ax.vlines(beat_positions_plotting, -0.9, 0.9, alpha=0.8, color='r',linestyle='dashed', linewidths=3, label='Beat Positions')
    ax.vlines(quarter_beat_positions, -0.7, 0.7, alpha=0.8, color='k',linestyle='dashed', linewidths=3, label='Quarter Beat Positions')
    
    ax.set_xlim([-0.05, (len(audio_array)/fs)+0.05])

    ax.set_title('Waveform', fontsize=20)
    ax.set_ylabel('Amplitude', fontsize=18)
    ax.set_xlabel('Time(s)', fontsize=18)      

    ax.tick_params(axis='both', which='major', labelsize=14)
    
def form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax):
    """
    Plots the bar, beat and quarter beats on a given spectrogram plt.ax
    """

    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(beat_positions)

    librosa.display.specshow(spectrogram, sr=fs, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax)

    ax.vlines(bar_positions, 0, 512, alpha=0.8, color='g',linestyle='dashed', linewidths=3, label='Bar Positions')
    ax.vlines(beat_positions_plotting, 0, 256, alpha=0.8, color='r',linestyle='dashed', linewidths=3, label='Beat Positions')
    ax.vlines(quarter_beat_positions, 0, 170, alpha=0.8, color='k',linestyle='dashed', linewidths=3, label='Quarter Beat Positions')
    
    ax.set_xlim([-0.05, (spectrogram.shape[1]*hop_length/fs)+0.05])
    ax.set_ylim([-5,512]) 

    ax.set_title('Spectrogram', fontsize=20, loc="center")
    ax.set_xlabel('Time(s)', fontsize=18)
    ax.set_ylabel('Hz', fontsize=18)    

    ax.yaxis.set_ticks(np.array([0,32,48,64,96,128,256,512]))
    ax.tick_params(axis='both', which='major', labelsize=14)

def form_pitch_track(F0_estimate, ax, color='b', label=''):
    """
    Plots the F0_estimate on a given plt.ax
    """

    time_axis, F0 = F0_estimate
    markerline, stemlines, baseline = ax.stem(time_axis, F0, basefmt=" ", label=label)
    markerline.set_markerfacecolor(color)
    markerline.set_markersize(9)
    stemlines.set_linewidth(0)

def create_sup_title(fig, title):
    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    fig.suptitle(title, fontsize=22, x=mid)    

# DEPRACATED
def form_notes(ax, notes, unk_notes):

    scale_notes = list(notes.keys())
    oos_notes =  list(unk_notes.keys())

    for i, note_dict in enumerate(list(notes.values())):
        if note_dict['time']:
            note = scale_notes[i]
            form_pitch_track((note_dict['time'], note_dict['frequency']), ax, color=colors[i], label=note)

    for j, note_dict in enumerate(list(unk_notes.values())):
        if note_dict['time']:
            note = oos_notes[j]
            form_pitch_track((note_dict['time'], note_dict['frequency']), ax, color=unk_colors[j], label='{}-OOS'.format(note))       

# DEPRACATED
def form_note_legend(ax, notes, unk_notes):
    """
    Formats the legend based on the number of notes
    """

    total_notes = len(notes.keys()) + len(unk_notes.keys())

    if total_notes > 10:
        ax.legend(loc=1, ncol=3, fontsize=15)
    elif total_notes > 6:
        ax.legend(loc=1, ncol=2, fontsize=15)
    else:
        ax.legend(loc=1, fontsize=15)

def save_function(plot_dir, track_title, plot_title='', default_title=''):
    """
    Saves the plot to a given directory with a given default plot title or the provided plot title.
    """

    os.makedirs(plot_dir, exist_ok=True)
    if not plot_title:
        plt.savefig(os.path.join(plot_dir,'{}-{}.jpeg'.format(track_title, default_title)))
    else:
        plt.savefig(os.path.join(plot_dir,'{}-{}.jpeg'.format(track_title, plot_title)))  