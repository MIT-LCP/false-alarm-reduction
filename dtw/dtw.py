from utils           import *
from parameters      import *
from datetime        import datetime
from scipy.spatial.distance import euclidean
import numpy         as np
import sklearn
import fastdtw
import wfdb
import json
import os

data_path = "../../sample_data/challenge_training_data/"
ann_path = "../../sample_data/challenge_training_multiann/"

def read_signals(data_path):
    signals_dict = {}
    fields_dict = {}

    for filename in os.listdir(data_path):
        if filename.endswith(HEADER_EXTENSION):
            sample_name = filename.rstrip(HEADER_EXTENSION)

            sig, fields = wfdb.rdsamp(data_path + sample_name)

            signals_dict[sample_name] = sig
            fields_dict[sample_name] = fields

    return signals_dict, fields_dict

def get_data(sig_dict, fields_dict, num_training):
    training_keys = list(sig_dict.keys())[:num_training]
    testing_keys = list(sig_dict.keys())[num_training:]

    sig_training = { key : sig_dict[key] for key in training_keys }
    fields_training = { key : fields_dict[key] for key in training_keys }
    sig_testing = { key : sig_dict[key] for key in testing_keys }
    fields_testing = { key : fields_dict[key] for key in testing_keys }

    return sig_training, fields_training, sig_testing, fields_testing


def sig_distance(sig1, fields1, sig2, fields2, radius=1, max_channels=1):
    channels_dists = {}
    channels1 = fields1['signame']
    channels2 = fields2['signame']

    common_channels = list(set(channels1).intersection(set(channels2)))
    if len(common_channels) > max_channels:
        common_channels = common_channels[:max_channels]

    start_index = int(FS * (ALARM_TIME-20))
    end_index = int(FS * ALARM_TIME)

    for channel in common_channels:
        if channel == "RESP":
            continue

        try:
            channel_index1 = channels1.index(channel)
            channel_index2 = channels2.index(channel)

        except Exception as e:
            print "channels1:", channels1, " channels2:", channels2, " common_channels:", common_channels, "\n", e
            continue

        channel1 = sig1[start_index:end_index,channel_index1]
        channel2 = sig2[start_index:end_index,channel_index2]

        try:
            distance, path = fastdtw.fastdtw(channel1, channel2, radius=radius, dist=euclidean)
        except Exception as e:
            # print "Error, continuing...", e
            continue

        channels_dists[channel] = distance

    return channels_dists


def normalize_distances(channels_dists, normalization='ecg_average', sigtypes_filename='../../sample_data/sigtypes'):
    if len(channels_dists.keys()) == 0:
        return float('inf')

    if len(channels_dists.keys()) == 1:
        return channels_dists.values().pop()

    ecg_channels = [ channel for channel in channels_dists if get_channel_type(channel, sigtypes_filename) == "ECG" ]
    ecg_dists = [ channels_dists[channel] for channel in ecg_channels ]

    if normalization == 'ecg_average':
        return np.mean(ecg_dists)

    if normalization == 'ecg_min':
        return min(ecg_dists)

    if normalization == 'ecg_max':
        return max(ecg_dists)

    if normalization == 'average':
        return np.mean(channels_dists.values())

    if normalization == 'min':
        return min(channels_dists.values())

    elif normalization == 'max':
        return max(channels_dists.values())

    raise Exception("Unrecognized normalization")


def predict(test_sig, test_fields, sig_training_by_arrhythmia, fields_training_by_arrhythmia):
    min_distance = float("inf")
    min_label = ""
    min_sample = ""

    arrhythmia = get_arrhythmia_type(test_fields)
    sig_training = sig_training_by_arrhythmia[arrhythmia]
    fields_training = fields_training_by_arrhythmia[arrhythmia]

    for sample_name, train_sig in sig_training.items():
        train_fields = fields_training[sample_name]
        channels_dists = sig_distance(test_sig, test_fields, train_sig, train_fields)
        distance = normalize_distances(channels_dists)

        # print "training sample:", sample_name, " distance:", distance, " min_distance:", min_distance

        if distance < min_distance:
            min_distance = distance
            min_label = is_true_alarm_fields(train_fields)
            min_sample = sample_name

    return min_label, min_distance, min_sample

## Get classification accuracy of testing based on training set
# sig_training_by_arrhythmia
def run_classification(sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing):
    num_correct = 0
    matrix = {
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": []
    }
    min_distances = {}

    for sample_name, test_sig in sig_testing.items():
        test_fields = fields_testing[sample_name]

        prediction, distance, sample = predict(test_sig, test_fields, sig_training_by_arrhythmia, fields_training_by_arrhythmia)
        actual = is_true_alarm_fields(test_fields)
        print "sample:", sample_name, " prediction:", prediction, " actual:", actual

        min_distances[sample_name] = (distance, sample, prediction == actual)

        if prediction and actual:
            matrix["TP"].append(sample_name)
        elif prediction and not actual:
            matrix["FP"].append(sample_name)
        elif not prediction and actual:
            matrix["FN"].append(sample_name)
        else:
            matrix["TN"].append(sample_name)

    return matrix, min_distances


def run(data_path, num_training, arrhythmias, matrix_filename, distances_filename):
    print "Generating sig and fields dicts..."
    sig_dict, fields_dict = read_signals(data_path)
    sig_training, fields_training, sig_testing, fields_testing = \
        get_data(sig_dict, fields_dict, num_training)
    sig_training_by_arrhythmia = { arrhythmia : get_samples_of_type(sig_training, arrhythmia) \
        for arrhythmia in arrhythmias }
    fields_training_by_arrhythmia = { arrhythmia : get_samples_of_type(fields_training, arrhythmia) \
        for arrhythmia in arrhythmias }

    print "Calculating classification accuracy..."
    matrix, min_distances = run_classification( \
        sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing)

    write_json(matrix, matrix_filename)
    write_json(min_distances, distances_filename)


def get_counts_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    return counts_by_arrhythmia


if __name__ == '__main__':
    num_training = 500
    arrhythmias = ['a', 'b', 't', 'v', 'f']
    # matrix_filename = "../../sample_data/dtw.json"
    matrix_filename = "../sample_data/pipeline_fpinvalids_vtachfpann.json"
    # distances_filename = "../../sample_data/dtw_distances.json"

    matrix = read_json(matrix_filename)
    # min_distances = read_json(distances_filename)
    counts = { key : len(matrix[key]) for key in matrix.keys()}
    
    print "accuracy:", get_classification_accuracy(matrix)
    print "score:", get_score(matrix)
    print_stats(counts)


    print "\nVTACH STATS"

    arrhythmia_dict = get_counts_by_arrhythmia(matrix, "v")
    arrhythmia_counts = { key : arrhythmia_dict[key][0] for key in arrhythmia_dict.keys() }
    arrhythmia_matrix = { key : arrhythmia_dict[key][1] for key in arrhythmia_dict.keys() }

    print "accuracy:", get_classification_accuracy(arrhythmia_matrix)
    print "score:", get_score(arrhythmia_matrix)
    print_stats(arrhythmia_counts)
