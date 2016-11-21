import numpy as np

##### Filename extensions #####
HEADER_EXTENSION = ".hea"
JQRS_EXTENSION = ".jqrs"

##### Constants #####

ALARM_TIME = 300 # in seconds
NUM_SECS_IN_MIN = 60
DEFAULT_FS = 250.0

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
MARKER_TYPES = [
    'o', 
    '^',
    's',
    '*',
    'x',
    'd'
]

##### Invalid sample detection constants #####

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
    "BP" : {
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
BOTTOM_PERCENTILE = 1
TOP_PERCENTILE = 99

##### Annotation constants #####
DEFAULT_ECG_FS = 250.0
DEFAULT_OTHER_FS = 125.0

##### Regular activity constants #####

TESTED_BLOCK_LENGTHS = { # seconds
    "Asystole": 14,
    "Bradycardia": 10, # 16,
    "Tachycardia": 10, # 14,
    "Ventricular_Tachycardia": 10,
    "Ventricular_Flutter_Fib": 13
}

RR_STDEV = 0.025 # seconds
HR_MIN = 45
HR_MAX = 135
MIN_NUM_RR_INTERVALS = 3
RR_MIN_SUM_DIFF = 2 # seconds
INVALIDS_SUM = 0

##### Specific arrhythmia tests #####
ASYSTOLE_WINDOW_SIZE = 3.2 # in seconds
ASYSTOLE_ROLLING_INCREMENT = 0.5 # in seconds

BRADYCARDIA_NUM_BEATS = 4
BRADYCARDIA_HR_MIN = 45

TACHYCARDIA_NUM_BEATS = 12
TACHYCARDIA_HR_MAX = 130

VTACH_NUM_BEATS = 4
VTACH_MAX_HR = 95
VTACH_ABP_THRESHOLD = 6 # in mmHg
VTACH_WINDOW_SIZE = 3 # in seconds
VTACH_ROLLING_INCREMENT = 0.5 # in seconds

VFIB_DLFMAX_LIMIT = 3 # in seconds
VFIB_LOW_DOMINANCE_INDEX_THRESHOLD = 100 # in sample number
VFIB_WINDOW_SIZE = 2 # in seconds
VFIB_ROLLING_INCREMENT = 0.5 # in seconds
VFIB_ABP_THRESHOLD = 6 # in mmHg
VFIB_DOMINANT_FREQ_THRESHOLD = 2 # in Hz 

##### Ventricular beat detection #####
LF_LOW = 1
LF_HIGH = 10
MF_LOW = 5
MF_HIGH = 25
HF_LOW = 50 
HF_HIGH = 70

VENTRICULAR_BEAT_THRESHOLD_RATIO = 0.5
VENTRICULAR_BEAT_PERCENTILE = 98
