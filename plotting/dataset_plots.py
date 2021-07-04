#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def key_pie_charts(m_counter, M_counter):

    fig, ax = plt.subplots(ncols=2, figsize=(16,7), constrained_layout=True)
    
    values = np.array(list(m_counter.values()))
    explode = np.zeros(12)
    explode[np.argpartition(values, -3)[-3:]] = [0.05, 0.1, 0.2]

    ax[0].set_title('Minor Keys Bassline Transcriptions', fontsize=15)
    ax[0].pie(values, explode=explode, labels=m_counter.keys(),textprops={'fontsize': 15})
    ax[0].axis('equal')

    values = np.array(list(M_counter.values()))
    explode = np.zeros(12)
    explode[np.argpartition(values, -3)[-3:]] = [0.05, 0.1, 0.2]

    ax[1].set_title('Major Keys Bassline Transcriptions', fontsize=15)
    ax[1].pie(values, explode=explode, labels=M_counter.keys(),textprops={'fontsize': 15})
    ax[1].axis('equal')
    plt.show()

def plot_note_occurances(note_counter, M, title=''):
    vals, labels = zip(*note_counter.items())
    min_orig, max_orig = min(vals), max(vals)
    print('Min: {}, max: {}'.format(min_orig, max_orig))
    fig, ax = plt.subplots(figsize=(15,5))
    fig.suptitle('MIDI Number Occurance Frequencies (M={})'.format(M), fontsize=22)
    ax.set_xlim([min_orig-1, max_orig+1])
    ax.bar(vals, labels)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.xaxis.set_tick_params(labelbottom=True)
    if title:
        plt.savefig('MIDI_number_distribution.jpg')
    plt.show()
    

def plot_note_occurances_with_transposing(note_counter, note_counter_T, M, title=''):

    vals, labels = zip(*note_counter.items())
    vals_T, labels_T = zip(*note_counter_T.items())
    
    min_orig, max_orig = min(vals), max(vals)
    min_T, max_T = min(vals_T), max(vals_T)
    print('Original min: {}, max: {}'.format(min_orig, max_orig))
    print('Transposed min: {}, max: {}'.format(min_T, max_T))

    fig, ax = plt.subplots(nrows=2, figsize=(20,10), sharex=True)
    fig.suptitle('MIDI Number Occurance Frequencies (M={})'.format(M), fontsize=22)

    ax[0].set_title('Without Transposing', fontsize=18)
    ax[0].set_xlim([min_orig-1, max_orig+1])
    ax[0].bar(vals, labels)
    ax[0].tick_params(axis='both', which='major', labelsize=15)
    ax[0].vlines([24, 48], 0, 100000, color='r', linewidth=5)
    ax[0].xaxis.set_tick_params(labelbottom=True)

    ax[1].set_title('After Transposing', fontsize=18)
    ax[1].set_xlim([min_T-1, max_T+1])
    ax[1].bar(vals_T, labels_T)
    ax[1].vlines([17, 36], 0, 100000, color='r', linewidth=5)
    ax[1].tick_params(axis='both', which='major', labelsize=15)
    if title:
        plt.savefig('MIDI_number_distribution.jpg')
    plt.show()