from utils                  import *
from parameters             import *
from datetime               import datetime
from scipy.signal           import resample
from scipy.spatial.distance import euclidean
from scipy.stats.mstats     import zscore
import numpy                as np
import sklearn
import fastdtw
import wfdb
import json
import os

data_path = "../sample_data/challenge_training_data/"
ann_path = "../sample_data/challenge_training_multiann/"

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


def sig_distance(sig1, fields1, sig2, fields2, radius, new_fs, max_channels=1, num_secs=10):
    channels_dists = {}
    channels1 = fields1['signame']
    channels2 = fields2['signame']

    common_channels = list(set(channels1).intersection(set(channels2)))
    if len(common_channels) > max_channels:
        common_channels = common_channels[:max_channels]

    start_index = int(FS * (ALARM_TIME-num_secs))
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

        # Downsample
        channel1_sampled = resample(channel1, num_secs*new_fs)
        channel2_sampled = resample(channel2, num_secs*new_fs)

        # Normalize
        channel1_normalized = zscore(channel1_sampled)
        channel2_normalized = zscore(channel2_sampled)

        try:
            if radius > 0: 
                distance, path = fastdtw.fastdtw(channel1_normalized, channel2_normalized, radius=radius, dist=euclidean)
            else: 
                distance = sum([val**2 for val in (channel1_normalized - channel2_normalized)])

        except Exception as e:
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


def predict(test_sig, test_fields, sig_training_by_arrhythmia, fields_training_by_arrhythmia, radius, new_fs, weighting):
    min_distance = float("inf")
    min_label = ""
    min_sample = ""

    arrhythmia = get_arrhythmia_type(test_fields)
    sig_training = sig_training_by_arrhythmia[arrhythmia]
    fields_training = fields_training_by_arrhythmia[arrhythmia]

    for sample_name, train_sig in sig_training.items():
        train_fields = fields_training[sample_name]
        channels_dists = sig_distance(test_sig, test_fields, train_sig, train_fields, radius, new_fs)
        distance = normalize_distances(channels_dists)

        if distance < min_distance:
            min_distance = distance
            min_label = is_true_alarm_fields(train_fields)
            min_sample = sample_name

    return min_label, min_distance, min_sample


## Get classification accuracy of testing based on training set
def run_classification(sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing, radius, new_fs, weighting):
    num_correct = 0
    matrix = {
        "TP": [],
        "FP": [],
        "TN": [],
        "FN": []
    }
    min_distances = {}

    for sample_name, test_sig in sig_testing.items():
        start = datetime.now()
        test_fields = fields_testing[sample_name]

        predicted, distance, sample = predict(test_sig, test_fields, sig_training_by_arrhythmia, fields_training_by_arrhythmia, radius, new_fs, weighting)
        actual = is_true_alarm_fields(test_fields)
        print "sample:", sample_name, " predicted:", predicted, " actual:", actual
        print "elapsed: ", datetime.now() - start

        min_distances[sample_name] = (distance, sample, predicted == actual)

        classification = get_matrix_classification(actual, predicted)
        matrix[classification].append(sample_name)

    return matrix, min_distances


def run(data_path, num_training, arrhythmias, matrix_filename, distances_filename, radius=0, new_fs=FS, weighting=1):
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
        sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing, radius, new_fs, weighting)

    write_json(matrix, matrix_filename)
    write_json(min_distances, distances_filename)


def get_counts_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    return counts_by_arrhythmia


if __name__ == '__main__':
    start = datetime.now()

    new_fs = 125
    num_training = 500
    arrhythmias = ['a', 'b', 't', 'v', 'f']
    matrix_filename = "../sample_data/dtw_radiusinf.json"
    # matrix_filename = "../sample_data/pipeline_fpinvalids_vtachfpann.json"
    distances_filename = "../sample_data/dtw_distances_radiusinf.json"

    run(data_path, num_training, arrhythmias, matrix_filename, distances_filename, radius=new_fs*2, new_fs=new_fs)

    # matrix = read_json(matrix_filename)
    # min_distances = read_json(distances_filename)

    # counts = { key : len(matrix[key]) for key in matrix.keys() }
    # print "accuracy:", get_classification_accuracy(matrix)
    # print "score:", get_score(matrix)
    # print_stats(counts)


    # print "\nVTACH STATS"
    # arrhythmia_dict = get_counts_by_arrhythmia(matrix, "v")
    # arrhythmia_counts = { key : arrhythmia_dict[key][0] for key in arrhythmia_dict.keys() }
    # arrhythmia_matrix = { key : arrhythmia_dict[key][1] for key in arrhythmia_dict.keys() }
    # print "accuracy:", get_classification_accuracy(arrhythmia_matrix)
    # print "score:", get_score(arrhythmia_matrix)
    # print_stats(arrhythmia_counts)
