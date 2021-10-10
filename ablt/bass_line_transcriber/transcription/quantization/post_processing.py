#!/usr/bin/env python
# coding: utf-8

import numpy as np

from .pitch_quantization import quantize_frequency


def onset_offset_merger(pitch_track, regions, length_threshold=8):
    """
    Merges the onset with the following segment and/or the offset with the previous segment.
    """

    F0 = pitch_track[1].copy()
    no_regions = len(regions)

    for region_idx, region_segments in enumerate(regions): # for each segment in the region

        no_segments = len(region_segments)

        if no_segments > 2: # straightforward merging
       
            for segment_idx, (start, end) in enumerate(region_segments): # get the segment boundaries

                segment_length = end - start

                if segment_length < length_threshold: # if the current segment has small length, apply merging

                    if segment_idx == 0: # onset merging

                        ns, _ = region_segments[segment_idx+1] # start idx of the next segment
                        f = F0[ns] # get the closest sample from the next segment
                        
                        F0[start:end] = f # replace the original samples

                    if segment_idx == no_segments-1: # offset merging

                        _, pe = region_segments[segment_idx-1] # an idx from the previous segment
                        f = F0[pe-1] # get the closest sample from the previous segment

                        F0[start:end] = f # replace the original samples

        elif no_segments == 2: #if there are 2 segments, ve take care of cases

            ps, pe = region_segments[0]
            ns, ne = region_segments[1]

            len1, len2 = pe-ps, ne-ns

            if not (len1==length_threshold and len2==length_threshold):

                if len1>len2:
                    f = F0[pe-1]
                    F0[ns : ne] = f
                else:
                    f = F0[ns]
                    F0[ps : pe] = f

        else: # for single segment it can not be length smaller than 8 anyway
            continue


    return (pitch_track[0], F0)
                            

def region_silencer(pitch_track, bad_regions):
    """
    Zeroes out given regions.
    """

    # get the indices to be silenced
    _, _, indices = bad_regions

    silenced_pitch_track = np.array([0.0 if idx in indices else f for idx, f in enumerate(pitch_track[1])])

    return (pitch_track[0], silenced_pitch_track)

# REMOVE!
def unk_filter(pitch_track, track_scale):

    _, scale_frequencies, _, out_of_scale_frequencies = track_scale

    F0 = pitch_track[1].copy()

    for idx, f in enumerate(F0):

        if f: # if f non-zero

            if f not in scale_frequencies:

                # look for close notes outside the scale
                f = quantize_frequency(f, out_of_scale_frequencies, 4)

                if f in out_of_scale_frequencies:
                    F0[idx] = f
                else:
                    F0[idx] = 0.0


    return (pitch_track[0], F0)