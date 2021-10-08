#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt

from .building_blocks import *


def spectrogram(title, beat_positions, spectrogram, fs, hop_length, F0_estimate=None, show=True, plot_dir='', plot_title=''):

    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)

    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)

    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax)

    if not (F0_estimate is None):
        form_pitch_track(F0_estimate, ax, label='pYIN Estimate')

    if plot_dir:
        save_function(plot_dir, title, plot_title=plot_title, default_title='Spectrogram')

    if show:
        plt.show()

def note_spectrogram(title, beat_positions, spectrogram, fs, hop_length, notes, unk_notes, show=True, plot_dir='', plot_title=''):

    fig, ax = plt.subplots(figsize=(20, 8), constrained_layout=True)
    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)

    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax)

    form_notes(ax, notes, unk_notes)

    form_note_legend(ax, notes, unk_notes)

    if plot_dir:
        save_function(plot_dir, title, plot_title=plot_title, default_title='Spectrogram_Notes')

    if show:
        plt.show()

def note_comparison_spectrogram(title, beat_positions, spectrogram, fs, hop_length, F0_estimate, notes, unk_notes, show=True, plot_dir='', plot_title=''):

    fig, ax = plt.subplots(figsize=(20, 10), nrows=2, sharex=False, constrained_layout=True)

    fig.suptitle(title+'\n\nIsolated Chorus Bassline in the Beat Grid ', fontsize=20)

    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax[0])
    form_notes(ax[0], notes, unk_notes)
    form_note_legend(ax[0], notes, unk_notes)

    form_beat_grid_spectrogram(beat_positions, spectrogram, fs, hop_length, ax[1])
    form_pitch_track(F0_estimate, ax[1], label='pYIN estimation')
    ax[1].legend(loc=1, fontsize=15)

    if plot_dir:
        save_function(plot_dir, title, plot_title=plot_title, default_title='Note_Comparison')

    if show:
        plt.show()