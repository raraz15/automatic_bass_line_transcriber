#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt

from .building_blocks import *

def waveform_and_spectrogram(title, beat_positions, audio_array, spectrogram, fs, hop_length, F0_estimate=None, show=True, plot_dir='', plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax[0])
    if F0_estimate:   
        form_pitch_track(F0_estimate, ax[0], label='F0 Estimate') 
    ax[0].legend(loc=1, fontsize=12)

    form_beat_grid_waveform(beat_positions, audio_array, fs, ax[1])

    if plot_dir:
        save_function(plot_dir, title, plot_title=plot_title, default_title='Wavefrom_and_Spectrogram')
    
    if show:
        plt.show()


def waveform_and_note_spectrogram(title, beat_positions, audio_array, spectrogram, fs, hop_length, notes, unk_notes, show=True, plot_dir='', plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)
    
    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax[0])
    form_notes(ax[0], notes, unk_notes)
    form_note_legend(ax[0], notes, unk_notes)

    form_beat_grid_waveform(beat_positions, audio_array, fs, ax[1])
    
    if plot_dir:
        save_function(plot_dir, title, plot_title=plot_title, default_title='Waveform_and_Note_Spectrogram')
   
    if show:   
        plt.show()
    else:
        plt.close('all')


def batch_plotting(title, directories, audio_array, spectrogram, fs, hop_length, F0_estimate=None, save=False, plot_title='', figsize=(20, 10)):
    
    fig, ax = plt.subplots(figsize=figsize, nrows=2, sharex=False, constrained_layout=True) #, dpi=50
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(title, directories, spectrogram, fs, hop_length, ax[0])
    
    if F0_estimate is not None:   
        form_pitch_track(F0_estimate, ax[0], label='F0 Estimate') 

    ax[0].legend(loc=1, fontsize=12)

    form_beat_grid_waveform(title, directories, audio_array, fs, ax[1])

    save_function(save, directories['plot']['wavefrom_spectrogram'], title, plot_title=plot_title, default_title='Waveform_and_Spectrogram')

    plt.close('all')
    plt.clf()
    
