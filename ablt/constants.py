# HERE WE DEFINE THE CONSTANTS THAT WE USE THROUGHOUT THE PROJECT

FS = 44100 # Sampling rate

# Some Sub-Bass Frequencies in Hz
F_B0 = 30.87  
F_C1 = 32.7 

F_B2 = 123.47 
F_C3 = 130.81

# Frequency Range of Interest
F_MIN = F_B0
F_MAX = F_C3

T_MAX = 1/F_MIN # Longest Period
FRAME_LEN = int(2*T_MAX*FS) # Frame Length

CUTOFF_FREQ = F_MAX # Post processing cut-off filter at the source separator

HOP_FACTOR = 32 # F0 estimation hop length during

M = 1 # Downsampling rate for symbolic representation creatinon
      # must be a power of 2 between 1 and HOP_FACTOR