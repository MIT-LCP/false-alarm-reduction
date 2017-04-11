import numpy as np

##### Filename extensions #####
HEADER_EXTENSION = ".hea"

##### Constants #####

ALARM_TIME = 300 # in seconds
FS = 250.0
DEFAULT_ECG_FS = 250.0
DEFAULT_OTHER_FS = 125.0


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
