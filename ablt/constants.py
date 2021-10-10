import numpy as np

# HERE WE DEFINE THE CONSTANTS THAT WE USE THROUGHOUT THE PROJECT

FS = 44100 # Sampling rate

# Some Sub-Bass Frequencies in Hz
#F_B0 = 30.87 # MIDI number 23
#F_C1 = 32.7  # MIDI number 24

#F_B2 = 123.47 # MIDI number 47
#F_C3 = 130.81 # MIDI number 48

# Frequency Range of Interest
MIDI_PITCH_MIN = 23
MIDI_PITCH_MAX = 48

PITCH_FREQUENCIES = np.around(440*np.power(2, (np.arange(0,128)-69)/12),2)
SUB_BASS_FREQUENCIES = PITCH_FREQUENCIES[MIDI_PITCH_MIN:MIDI_PITCH_MAX+1]

F_MIN = SUB_BASS_FREQUENCIES[0]
F_MAX = SUB_BASS_FREQUENCIES[-1]

T_MAX = 1/F_MIN # Longest Period

FRAME_FACTOR = 2 # Number of periods that make an F0 estimation frame

FRAME_LEN = int(FRAME_FACTOR*T_MAX*FS) # Frame Length

CUTOFF_FREQ = F_MAX # Post processing cut-off filter at the source separator

HOP_RATIO = 32 # F0 estimation hop length wrt. a beat

M = 1 # Downsampling rate for symbolic representation creatinon
      # must be a power of 2 between 1 and HOP_RATIO

DROP_DETECTOR_CUTOFF = 128 # Cutoff frequency for drop detection 