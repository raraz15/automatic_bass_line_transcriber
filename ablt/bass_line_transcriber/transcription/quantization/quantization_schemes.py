#!/usr/bin/env python
# coding: utf-8

import numpy as np

from .pitch_quantization import uniform_quantization
from .segmentation import find_voiced_regions, segment_voiced_regions, get_region_information
from .post_processing import onset_offset_merger, region_silencer


def uniform_voiced_region_quantization(pitch_track, track_scale, epsilon=2):
    """
    Finds the voiced regions, and uniformly quantizes each region in frequency using majority voting.

        Parameters:
        -----------

            pitch_track (tupple): (time_axis, F0) where both are np.ndarray
            track_scale (tupple): (notes, scale_frequencies, out_notes, out_frequencies)
            epsilon (int, default=4): freq_bound = delta_scale/epsilon determines if quantization will happen.

        Returns:
        --------

            pitch_track_quantized (tupple): (time_axis, F0) where both are np.ndarray
    """

    voiced_regions = find_voiced_regions(pitch_track[1])   

    pitch_track_quantized = uniform_quantization(pitch_track, track_scale[1], voiced_regions, epsilon)

    return  pitch_track_quantized


def adaptive_voiced_region_quantization(pitch_track, quarter_beat_positions, length_threshold=8, epsilon=2):
    """
    Aplies adaptive quantization to voiced regions accoring to the algorithm 1.

        Parameters:
        -----------
        
            pitch_track (tupple): (time_axis, F0) where both are np.ndarrays
            quarter_beat_positions (ndarra): ndarray of quarter beat time values
            length_threshold (int, default=8): determines short regions to filter out
            epsilon (int, default=4): freq_bound = delta_scale/epsilon determines if quantization will happen.

        Returns:
        --------

            pitch_track_quantized (tupple): (time_axis, F0) where both are np.ndarrays
    """

    # Find the voiced regions
    voiced_boundaries, _, _ = find_voiced_regions(pitch_track[1])

    # segment the voiced regions
    segmented_good_regions, okay_regions, bad_regions = segment_voiced_regions(pitch_track[0], 
                                                                        voiced_boundaries,
                                                                        length_threshold,
                                                                        quarter_beat_positions)

    # flatten the boundaries, and create segment tupple = (bounds, lens, indices)
    good_regions = get_region_information(np.array([bounds for region in segmented_good_regions for bounds in region]))

    # Uniformly quantize each segmented region independtly 
    pitch_track_quantized = uniform_quantization(pitch_track, good_regions, epsilon)

    # Merge the onsets and the offsets using segmented_good_regions
    pitch_track_quantized = onset_offset_merger(pitch_track_quantized, segmented_good_regions)

    # Uniformly quantize okay regions, without segmentation
    pitch_track_quantized = uniform_quantization(pitch_track_quantized, okay_regions, epsilon)

    # Silence bad regions
    pitch_track_quantized = region_silencer(pitch_track_quantized, bad_regions)

    return pitch_track_quantized