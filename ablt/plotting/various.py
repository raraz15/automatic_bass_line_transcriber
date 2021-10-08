#!/usr/bin/env python
# coding: utf-8

import os

import numpy as np
from matplotlib import pyplot as plt

from .building_blocks import *

# TODO: needs cleaning and fixing functions

colors = ['0.5','tab:orange','tab:olive','moccasin','khaki','steelblue','b','g','r','c','m','y','k','c','w']
unk_colors = ['purple','hotpink','lime','firebrick','salmon','darkred','mistyrose']

root_dir = '/mnt/d/projects/bassline_extraction'
plot_dir = os.path.join(root_dir, 'figures', 'plots')

confidence_dir = os.path.join(plot_dir, 'F0 Confidence Plots')
spectral_plots_dir = os.path.join(plot_dir, 'Spectral Plots')

spectral_comparison_dir = os.path.join(spectral_plots_dir, 'comparisons')

confidence_filtering_spec_dir = os.path.join(spectral_comparison_dir, 'confidence_filtering')
algorithm_comparison_spec_dir = os.path.join(spectral_comparison_dir, 'algorithm_comparison')

algorithm_comparison_raw_dir = os.path.join(algorithm_comparison_spec_dir, 'raw_outputs')
algorithm_comparison_confidence_dir = os.path.join(algorithm_comparison_spec_dir, 'confidence_filtered')
algorithm_comparison_quantized_dir = os.path.join(algorithm_comparison_spec_dir, 'quantized')


def plot_confidence_filtering_effect(title, spectrogram, fs, hop_length, F0_estimate, pitch_track, save=False, plot_title=''):

    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    ax[0].set_title('Initial Estimation', fontsize=16)
    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimate, ax[0])

    ax[1].set_title('Confidence Level Filtered', fontsize=16)
    form_beat_grid_spectrogram(title, ax[1], spectrogram, fs, hop_length)
    form_pitch_track(pitch_track, ax[1]) 

    save_function(save, confidence_filtering_spec_dir, title, plot_title=plot_title, default_title='Confidence_Filtering')

    plt.show()


def plot_algorithm_comparison_raw(title, spectrogram, fs, hop_length, F0_estimates, estimator_names, save=False, plot_title=''):
    """
    Plots the comparison of two F0 estimator algorithm's raw outputs.
    """
    
    bar_positions, beat_positions_plotting, quarter_beat_positions = beat_plotting(title)
    
    fig, ax = plt.subplots(figsize=(20,10), nrows=2, sharex=False, constrained_layout=True)
    fig.suptitle(title, fontsize=20)

    form_beat_grid_spectrogram(title, ax[0], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimates[0], ax[0])

    form_beat_grid_spectrogram(title, ax[1], spectrogram, fs, hop_length)
    form_pitch_track(F0_estimates[1], ax[1])

    save_function(save, algorithm_comparison_raw_dir, title, plot_title=plot_title, default_title='Raw_Outputs')
              
    plt.show()


def plot_confidence(title, confidence, save=False, plot_title=''):

    histogram, bin_edges = np.histogram(confidence, bins=50)

    fig, ax = plt.subplots()

    width = 0.7 * (bin_edges[1] - bin_edges[0])
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(center, histogram, align='center', width=width, label='Histogram')
    ax.vlines(confidence.mean(), 0, histogram.max()+20, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax.vlines(confidence.mean()+confidence.std()*np.array([-1, 1])/2, 0, histogram.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax.set_title('{} - ({})'.format('pYIN',title))
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Number of Occurances')
    
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1))
    
    ax.legend()

    save_function(save, confidence_dir, title, plot_title=plot_title, default_title='Confidence_Level')

    plt.close()
    #plt.show()


def plot_compared_confidences(title, confidence_crepe, confidence_pyin, save=False):

    histogram_crepe, bin_edges_crepe = np.histogram(confidence_crepe, bins=25)
    histogram_pyin, bin_edges_pyin = np.histogram(confidence_pyin, bins=25)

    fig, ax = plt.subplots(2, 1, figsize=(10,8), sharex=True, sharey=True,constrained_layout=True)

    fig.suptitle('{} Confidence Level Comparisons'.format(title))

    width = 0.7 * (bin_edges_crepe[1] - bin_edges_crepe[0])
    center = (bin_edges_crepe[:-1] + bin_edges_crepe[1:]) / 2
    ax[0].bar(center, histogram_crepe, align='center', width=width, label='Histogram')
    ax[0].vlines(confidence_crepe.mean(), 0, histogram_crepe.max()+50, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax[0].vlines(confidence_crepe.mean()+confidence_crepe.std()*np.array([-1, 1])/2, 0, histogram_crepe.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax[0].set_title('CREPE')
    ax[0].set_xlabel('Confidence')
    ax[0].set_ylabel('#Occurances')
    ax[0].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    ax[0].legend()

    width = 0.7 * (bin_edges_pyin[1] - bin_edges_pyin[0])
    center = (bin_edges_pyin[:-1] + bin_edges_pyin[1:]) / 2
    ax[1].bar(center, histogram_pyin, align='center', width=width, label='Histogram')
    ax[1].vlines(confidence_pyin.mean(), 0, histogram_pyin.max()+50, color='r', linewidth=4, linestyle='dashed', label='Mean')
    ax[1].vlines(confidence_pyin.mean()+confidence_pyin.std()*np.array([-1, 1])/2, 0, histogram_pyin.max()-50, 
              color='k', linewidth=4, linestyle='dashed', label='0.5 STD')
    ax[1].set_title('pYIN')
    ax[1].set_xlabel('Confidence')
    ax[1].set_ylabel('#Occurances')
    ax[1].tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
    ax[1].legend()  


    if save:
        plt.savefig(os.path.join(confidence_dir, '{}_confidence_comparisons.png'.format(title)))
        plt.close()
    else:
        plt.show()


def energy_levels(title, energies, possible_drops, estimated_drop, epsilon=1, show=True, save=False, plot_title=''):
    
    no_half_sections = int( len(energies)/8 + 1)
    no_sections = int(no_half_sections / 2)
    
    pre_drop_energies = [energies[idx-1] for idx in possible_drops[0]]
    drop_energies = [energies[idx] for idx in possible_drops[0]]

    mean, std = np.mean(energies), np.std(energies)
    threshold = mean - (std/epsilon)
    
    fig, ax = plt.subplots(figsize=(20,10))

    markerline, stemlines, baseline = ax.stem(range(len(energies)), energies, basefmt=" ", label='Energy Levels')
    markerline.set_markerfacecolor('b')
    markerline.set_markersize(8)
    stemlines.set_linewidth(0)

    ax.vlines(8*np.arange(no_half_sections), 0, max(energies)+1700, alpha=0.8, color='c',linestyle='dashed', linewidths=3)
    ax.hlines(mean, 0, len(energies), color='g', linewidth=4, label='Average Energy')
    ax.hlines(threshold, 0, len(energies), color='r', linewidth=4, label='Energy Threshold')
    ax.vlines(possible_drops[0], pre_drop_energies, drop_energies, color='k', linewidth=5, label='Possible Drops')
    ax.vlines(estimated_drop[0], energies[estimated_drop[0]-1], energies[estimated_drop[0]], color='m', linewidth=5, label='Chosen Drop')

    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Bars', fontsize=16)
    ax.set_ylabel('Square Amplitude', fontsize=15)
    ax.set_xticks(16*np.arange(no_sections))
    ax.tick_params(axis='x', labelsize=14)

    ax.legend(loc=1, ncol=2, fontsize=14)
    
    save_function(save, os.path.join(root_dir,'figures','energy_levels'), title, plot_title, default_title='')

    if show:
        plt.show()