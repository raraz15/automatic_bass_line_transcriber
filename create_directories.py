import os
import json

from utilities import init_folders

if __name__ == "__main__":

    project_dir = os.getcwd()
    
    # directory to contain all the data
    data_dir = os.path.join(project_dir, 'data') 

    figures_dir = os.path.join(data_dir, 'figures')
    metadata_dir = os.path.join(data_dir, 'metadata') 
    clip_dir = os.path.join(data_dir, 'audio_clips')

    for dir in [figures_dir, metadata_dir, clip_dir]:
	    os.makedirs(dir, exist_ok=True)

    # Extraction Related Directories
    bassline_extraction_dir = os.path.join(data_dir,'bassline_extraction')

    beat_grid_dir = os.path.join(bassline_extraction_dir,'beat_grid')
    beat_grid = {'beat_grid': beat_grid_dir,
		        'beat_positions': os.path.join(beat_grid_dir, 'beat_positions')}

    chorus_dir = os.path.join(bassline_extraction_dir, 'chorus')
    chorus = {'chorus': chorus_dir,
	    'chorus_start_beat_idx': os.path.join(chorus_dir, 'chorus_start_beat_idx'),
	    'chorus_beat_positions': os.path.join(chorus_dir, 'chorus_beat_positions'),
	    'chorus_beat_analysis': os.path.join(chorus_dir, 'chorus_beat_analysis'),
	    'chorus_array': os.path.join(chorus_dir, 'chorus_array')}

    bassline_dir = os.path.join(bassline_extraction_dir, 'bassline')

    exception_logs_dir = os.path.join(bassline_extraction_dir, 'exceptions')

    extraction = {'clip': clip_dir,
		'bassline_extraction': bassline_extraction_dir,
		'beat_grid': beat_grid,
		'chorus': chorus,
		'bassline': bassline_dir,
		'exceptions': exception_logs_dir,
		'metadata': metadata_dir}
  

    # Transcription Related Directories
    bassline_transcription_dir = os.path.join(data_dir, 'bassline_transcription')

    trans ={'bassline_transcription': bassline_transcription_dir,
	    'F0_estimate': os.path.join(bassline_transcription_dir, 'F0_estimate'),
	    'pitch_track': os.path.join(bassline_transcription_dir, 'pitch_track'),
	    'quantized_pitch_track': os.path.join(bassline_transcription_dir, 'quantized_pitch_track')}


    representation_dir = os.path.join(bassline_transcription_dir, 'symbolic_representation')
    # merges
    representation = {'symbolic_representation': representation_dir}
    representation = {**representation, **{d: os.path.join(representation_dir ,d) for d in ['1','2','4','8']}}

    # Experiment Logs
    exception_logs_dir = os.path.join(bassline_transcription_dir, 'exceptions')
    
    transcription ={'bassline_transcription': trans,
		    'symbolic_representation': representation,
		    'exceptions': exception_logs_dir,
		    'metadata': metadata_dir}                    

    plot_dir = os.path.join(figures_dir, 'plots')

    spectral_plots_dir = os.path.join(plot_dir, 'Spectral Plots')

    spectrogram_dir = os.path.join(spectral_plots_dir, 'spectrograms')
    note_spec_dir = os.path.join(spectral_plots_dir, 'notes')
    spectral_comparison_dir = os.path.join(spectral_plots_dir, 'comparisons')

    time_freq_dir = os.path.join(plot_dir, 'Time-Frequency Plots')

    wave_spec_dir = os.path.join(time_freq_dir, 'wave_spectrograms')
    note_wave_spec_dir = os.path.join(time_freq_dir, 'notes')

    plot = {'figures': figures_dir,
	    'plot': plot_dir,
	    'spectral_plots': spectral_plots_dir,
	    'spectrogram': spectrogram_dir,
	    'note_spectrogram': note_spec_dir,
	    'spectral_comparison': spectral_comparison_dir,
	    'time_freq': time_freq_dir,
	    'wavefrom_spectrogram': wave_spec_dir,
	    'waveform_and_note_spectrogram': note_wave_spec_dir}

    # Midi 
    midi_dir = os.path.join(data_dir, 'midi')

    midi_array_dir = os.path.join(midi_dir, 'midi_array')
    midi_array = {'midi_array': midi_array_dir}
    midi_array = {**midi_array, **{d: os.path.join(midi_dir, 'midi_array',d) for d in ['1','2','4','8']}}

    midi_file_dir = os.path.join(midi_dir, 'midi_file')
    midi_file = {'midi_file': midi_file_dir}
    midi_file = {**midi_file, **{d: os.path.join(midi_dir, 'midi_file',d) for d in ['1','2','4','8']}}

    generated_dir = os.path.join(midi_dir, 'generated')
    generated = {'generated': generated_dir}
    generated = {**generated, **{d: os.path.join(midi_dir, 'generated',d) for d in ['1','2','4','8']}}

    midi = {'midi': midi_dir,
	    'midi_array': midi_array,
	    'midi_file': midi_file,
	    'generated': generated}

    # solve dependencies
    transcription['midi'] = midi 

    directories = {'extraction': extraction,
                    'transcription': transcription,
                    'midi': midi,
                    'metadata': metadata_dir,
                    'plot': plot}         

    init_folders(directories)                           

    with open(os.path.join(data_dir ,'directories.json'), 'w') as outfile:
	    json.dump(directories, outfile, indent=4)                  