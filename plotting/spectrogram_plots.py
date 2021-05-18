#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from matplotlib import pyplot as plt

from .building_blocks import *

#get_ipython().run_line_magic('matplotlib', 'inline') # ?????????????????


def spectrogram(title, directories, spectrogram, fs, hop_length, F0_estimate=None, show=True, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)
    
    form_beat_grid_spectrogram(title, directories, ax, spectrogram, fs, hop_length)
    
    if not (F0_estimate is None):
        form_pitch_track(F0_estimate, ax, label='pYIN Estimate')

    save_function(save, directories['plot']['spectrogram'], title, plot_title=plot_title, default_title='Spectrogram')
               
    if show:              
        plt.show() 


def note_spectrogram(title, directories, spectrogram, fs, hop_length, notes, unk_notes, show=True, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,8), constrained_layout=True)
    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)
    
    form_beat_grid_spectrogram(title, directories, ax, spectrogram, fs, hop_length)

    form_notes(ax, notes, unk_notes)

    form_note_legend(ax, notes, unk_notes) 

    save_function(save, directories['plot']['note_spectrogram'], title, plot_title=plot_title, default_title='Spectrogram_Notes')  

    if show:              
        plt.show()     
    
    
def note_comparison_spectrogram(title, directories, spectrogram, fs, hop_length, F0_estimate, notes, unk_notes, show=True, save=False, plot_title=''):
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)
    

    form_beat_grid_spectrogram(title, directories, ax[0], spectrogram, fs, hop_length)

    form_notes(ax[0], notes, unk_notes)

    form_note_legend(ax[0], notes, unk_notes) 


    form_beat_grid_spectrogram(title, directories, ax[1], spectrogram, fs, hop_length)
    
    form_pitch_track(F0_estimate, ax[1], label='pYIN estimation')

    ax[1].legend(loc=1, fontsize=15)

    
    save_function(save, directories['plot']['spectral_comparison'], title, plot_title=plot_title, default_title='Note_Comparison')  
    
    if show:              
        plt.show()    



# REDUNDANT ?
def compare_spectrograms(title, spectrogram0, spectrogram1, fs, hop_length, notes, unk_notes):

    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)
    

    form_beat_grid_spectrogram(title, ax[0], spectrogram0, fs, hop_length)

    form_notes(ax[0], notes, unk_notes)

    form_note_legend(ax[0], notes, unk_notes) 


    form_beat_grid_spectrogram(title, ax[1], spectrogram1, fs, hop_length)

    form_notes(ax[1], notes, unk_notes)

    form_note_legend(ax[1], notes, unk_notes)

    plt.show()      