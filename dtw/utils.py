

def abs_value(x, y):
    return abs(x-y)


def is_true_alarm(fields):
    return fields['comments'][1] == 'True alarm'


## Returns type of arrhythmia alarm
# output types include: 'a', 'b', 't', 'v', 'f'
def get_arrhythmia_type(fields):
    """Returns type of arrhythmia based on fields of the sample

    Arguments
    ---------
    fields: fields of sample read from wfdb.rdsamp

    Returns
    -------
    Type of arrhythmia
        'a': asystole
        'b': bradycardia
        't': tachycardia
        'f': ventricular fibrillation
        'v': ventricular tachycardia
    """

    arrhythmias = {
        'Asystole': 'a',
        'Bradycardia': 'b',
        'Tachycardia': 't',
        'Ventricular_Tachycardia': 'v',
        'Ventricular_Flutter_Fib': 'f'
    }

    arrhythmia_type = fields['comments'][0]
    return arrhythmias[arrhythmia_type]


def get_channel_type(channel_name, sigtypes_filename):
    """Returns type of channel

    Arguments
    ---------
    channel_name: name of channel (e.g. "II", "V", etc.)

    sigtypes_filename: file mapping channel names to channel
    types

    Returns
    -------
    Type of channel (e.g. "ECG", "BP", "PLETH", "Resp")
    """

    channel_types_dict = {}
    with open(sigtypes_filename, "r") as f:
        for line in f:
            splitted_line = line.split("\t")
            channel = splitted_line[-1].rstrip()
            channel_type = splitted_line[0]
            channel_types_dict[channel] = channel_type

    if channel_name in channel_types_dict.keys():
        return channel_types_dict[channel_name]

    raise Exception("Unknown channel name")


def get_samples_of_type(samples_dict, arrhythmia_type):
    """Returns a sub-dictionary of only the given arrhythmia type

    Arguments
    ---------
    samples_dict: dictionary mapping sample names to data associated
    with the given sample

    arrhythmia_type:
        'a': asystole
        'b': bradycardia
        't': tachycardia
        'f': ventricular fibrillation
        'v': ventricular tachycardia

    Returns
    -------
    a sub-dictionary with keys of only the given arrhythmia
    """

    subdict = {}

    for sample_name in samples_dict.keys():
        if sample_name[0] == arrhythmia_type:
            subdict[sample_name] = samples_dict[sample_name]

    return subdict
