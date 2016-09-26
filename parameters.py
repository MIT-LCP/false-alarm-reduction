import numpy as np

##### Constants #####

# Hard-coded colors 
COLORS = np.asarray(
    [[55,126,184],
    [228,26,28],
    [77,175,74],
    [152,78,163],
    [255,127,0],
    [255,255,51],
    [166,86,40],
    [247,129,191]])/256.0

# Plot marker types
MARKER_TYPES = ['o', '^']

##### Invalid sample detection constants #####

FS = 250
BLOCK_LENGTH = 0.8

ORDER = 50
F_LOW = 70
F_HIGH = 90
HIST_CUTOFF = 150
AMPL_CUTOFF = 0.005
STATS_CUTOFFS = {
    "ECG" : {
        "val_min" : -7,
        "val_max" : 7, 
        "var_range_min" : 0.005,
        "var_range_max" : 8
    }, 
    "ABP" : {
        "val_min" : 0,
        "val_max" : 300, 
        "var_range_min" : 0.0001,
        "var_range_max" : 250
    }, 
    "PLETH" : {
        "val_min" : -6,
        "val_max" : 6, 
        "var_range_min" : 0.005,
        "var_range_max" : 7
    }
}


##### Regular activity constants #####

RR_STDEV = 0.025
HR_MIN = 45
HR_MAX = 135
MIN_NUM_RR_INTERVALS = 3
