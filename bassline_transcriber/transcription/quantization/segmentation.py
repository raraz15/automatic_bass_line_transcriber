#!/usr/bin/env python
# coding: utf-8

import numpy as np


def calcRegionBounds(bool_array):
    '''
    Returns the lower and upper bounds of contiguous regions.
    Upper bound is not included in the region i.e [start, end)

    Parameters
    ==========
    bool_array  1-D Binary numpy array
    '''
    assert(bool_array.dtype == 'bool' )
    idx = np.diff(np.r_[0, bool_array, 0]).nonzero()[0]
    assert(len(idx)%2 == 0)
    return np.reshape(idx, (-1,2))


def find_voiced_regions(F0):
    """
    From a given F0 array, finds the voiced regions' boundaries and returns them with corresponding lengths and indices.
    """
    
    voiced_boundaries = calcRegionBounds(F0 != 0.0)
    
    return get_region_information(voiced_boundaries)


def get_region_information(boundaries):
    """
    Packs the boundaries, lengths, and the corresponding indices in a tupple.
    """

    lengths = np.diff(boundaries, 1).flatten().tolist()

    indices = [x for start, end in boundaries for x in np.arange(start, end)]

    return (boundaries, lengths, indices)


def find_closest_quarter_beat(time, quarter_beat_positions):
    
    delta = np.abs(time - quarter_beat_positions)
    delta_min = np.min(delta)
    idx = np.where(delta==delta_min)[0][0]
        
    return idx, quarter_beat_positions[idx]


def find_closest_note(time_axis, beat_time):
    
    delta = np.abs(beat_time - time_axis)
    delta_min = np.min(delta)
    
    return np.where(delta==delta_min)[0][0]    


def segment_voiced_regions(time_axis, region_boundaries, length_threshold, quarter_beat_positions):
    """
    Segments voiced regions if they have proper length, otherwise categorizes them.

        Parameters:
        -----------
        time_axis (ndarray): time axis array
        region_boundaries ():
        length_threshold (int): the threshold in deciding if region length is suitable for segmentation
    """
    
    quarter_beat_positions -= quarter_beat_positions[0] # start from time 0

    delta_time = np.diff(time_axis).max()/2 # maximum allowed distance deviation for beatgrid from the notes

    segmented_good_regions = [] # good regions 
    okay_region_boundaries = [] # okayish regions
    bad_region_boundaries = []  # bad regions

    # for each voiced region
    for onset_idx, upper_bound in region_boundaries:

        offset_idx = upper_bound - 1 # upper bound is excluded

        region_length = offset_idx - onset_idx + 1 

        if region_length > length_threshold: # if region length is suitable for segmentation

            segment_boundaries = [] # segmentation boundaries


            # get the times. Upper bounds of regions are excluded
            onset_time, offset_time = time_axis[onset_idx], time_axis[offset_idx]
            
            # find the closest quarter beats to the onset and offset times 
            start_beat_idx, start_beat_time = find_closest_quarter_beat(onset_time, quarter_beat_positions) 
            end_beat_idx, end_beat_time = find_closest_quarter_beat(offset_time, quarter_beat_positions)

            # make sure that onset starts before a qbeat and the offest ends after a qbeat
            if onset_time > start_beat_time and delta_time <=  onset_time - start_beat_time:
                start_beat_idx += 1
                start_beat_time = quarter_beat_positions[start_beat_idx]

            if offset_time < end_beat_time and delta_time <= end_beat_time - offset_time :
                end_beat_idx -= 1
                end_beat_time = quarter_beat_positions[end_beat_idx]


            # segmentation starts with the onset idx and the closest quarter beat's corresponding idx    
            b1 = onset_idx
            b2 = find_closest_note(time_axis, start_beat_time)
            if not b1 == b2: # onset on quarter beat
                segment_boundaries.append([b1, b2])
                
            # segment between each qbeat inside the adjusted region 
            for k in np.arange(start_beat_idx, end_beat_idx):

                # get the qbeat times
                start_beat_time = quarter_beat_positions[k] 
                end_beat_time = quarter_beat_positions[k+1]

                # get the corresponding indices in the pitch track
                b1 = find_closest_note(time_axis, start_beat_time)
                b2 = find_closest_note(time_axis, end_beat_time)

                segment_boundaries.append([b1, b2])
                
            # segmentation finishes with the final qbeat and the upper_bound = offset idx+1
            b1 = find_closest_note(time_axis, quarter_beat_positions[end_beat_idx])
            b2 = upper_bound # upper bounds are excluded ()
            if not b1 == b2: # offset on quarter beat
                segment_boundaries.append([b1, b2])

            segmented_good_regions.append(segment_boundaries)

        elif region_length >= int(length_threshold/2): # if the region can be uniformly quantized

            okay_region_boundaries.append([onset_idx, upper_bound])

        else: # if the region will be erased
               
            bad_region_boundaries.append([onset_idx, upper_bound])

    okay_regions = get_region_information(np.array(okay_region_boundaries))
    bad_regions = get_region_information(np.array(bad_region_boundaries))

    return segmented_good_regions, okay_regions, bad_regions