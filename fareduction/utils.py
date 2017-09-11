import wfdb
import json

def abs_value(x, y):
    return abs(x-y)

def is_true_alarm_fields(fields):
    return fields['comments'][1] == 'True alarm'


def is_true_alarm(data_path, sample_name):
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    return is_true_alarm_fields(fields)

# start and end in seconds
def get_annotation(sample, ann_type, ann_fs, start, end): 
    try: 
        annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
    except Exception as e: 
        annotation = []
        print(e)
    
    return annotation

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


def write_json(dictionary, filename):
    with open(filename, "w") as f:
        json.dump(dictionary, f)


def read_json(filename):
    with open(filename, "r") as f:
        dictionary = json.load(f)
    return dictionary


def get_classification_accuracy(matrix):
    num_correct = len(matrix["TP"]) + len(matrix["TN"])
    num_total = len(matrix["FP"]) + len(matrix["FN"]) + num_correct

    return float(num_correct) / num_total


def calc_sensitivity(counts): 
    tp = counts["TP"]
    fn = counts["FN"]
    return tp / float(tp + fn)
    

def calc_specificity(counts): 
    tn = counts["TN"]
    fp = counts["FP"]
    
    return tn / float(tn + fp)


def calc_ppv(counts): 
    tp = counts["TP"]
    fp = counts["FP"]
    return tp / float(tp + fp)


def calc_npv(counts): 
    tn = counts["TN"]
    fn = counts["FN"]
    return tn / float(tn + fn)


def calc_f1(counts): 
    sensitivity = calc_sensitivity(counts)
    ppv = calc_ppv(counts)
    
    return 2 * sensitivity * ppv / float(sensitivity + ppv)    


def print_stats(counts): 
    try: 
        sensitivity = calc_sensitivity(counts)
        specificity = calc_specificity(counts)
        ppv = calc_ppv(counts)
        npv = calc_npv(counts)
        f1 = calc_f1(counts)
    except Exception as e: 
        print(e)

    print("counts: ", counts)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("ppv: ", ppv)
    print("npv: ", npv)
    print("f1: ", f1)


def get_matrix_classification(actual, predicted): 
    if actual and predicted: 
        return "TP"
    elif actual and not predicted: 
        return "FN"
    elif not actual and predicted: 
        return "FP"
    return "TN"


def get_score(matrix):
    numerator = len(matrix["TP"]) + len(matrix["TN"])
    denominator = len(matrix["FP"]) + 5*len(matrix["FN"]) + numerator

    return float(numerator) / denominator


def get_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    matrix_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = len(sample_list)
        matrix_by_arrhythmia[classification_type] = sample_list

    return counts_by_arrhythmia, matrix_by_arrhythmia
