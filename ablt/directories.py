import os

# HERE WE DEFINE THE DIRECTORIES THROUGHOUT THE PROJECT

LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(LIBRARY_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
AUDIO_DIR = os.path.join(DATA_DIR, 'audio_clips')
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')
FIGURES_DIR = os.path.join(DATA_DIR, 'figures')

TRACK_DICTS_PATH = os.path.join(METADATA_DIR, "track_dicts.json")
SCALE_FREQUENCIES_PATH = os.path.join(METADATA_DIR, "scales_frequencies.json")