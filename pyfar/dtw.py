from __future__ import print_function

from utils                  import *
from parameters             import *
from datetime               import datetime
from scipy.signal           import resample
from scipy.spatial.distance import euclidean
from scipy.stats.mstats     import zscore
import numpy                as np
import matplotlib.pyplot    as plt
import sklearn
import fastdtw
import wfdb
import json
import os
import glob
import csv

def read_signals(data_path):
    signals_dict = {}
    fields_dict = {}

    for filename in os.listdir(data_path):
        if filename.endswith(HEADER_EXTENSION):
            sample_name = filename.rstrip(HEADER_EXTENSION)

            sig, fields = wfdb.srdsamp(data_path + sample_name)

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


def downsample_signal(sig, fields, Fnew=125):
    Fs = fields['fs']

    # downsample if needed
    if Fnew<Fs:
        if int(Fs/Fnew)==(Fs/Fnew):
            sig_new = resample(sig, len(sig)/int(Fs/Fnew))
    elif Fnew>Fs:
        sig_new = sig
        print('{} is higher than sampling frequency ({}) - not resampling.'.format(Fnew, Fs))
    return sig_new


def alt_dtw():
    T_START=290
    T_END=300
    Fnew=125

    for s in fields_dict:
        print("\n\n" + s)
        for j, lead in enumerate(fields_dict[s]['signame']):
            # get signal lead name
            if lead in ['I','II','III','V']:
                # print('\t' + lead, end=' ')
                sig1 = (np.copy(signals_dict[s][:,j])*fields_dict[s]['gain'][j]).astype(int)

                # downsample the signal
                sig1 = downsample_signal(sig1, fields_dict[s], Fnew=Fnew)

                # extract the 10 seconds of interest
                sig1 = sig1[T_START*Fnew:T_END*Fnew]

                # normalize
                mu1 = np.mean(sig1)
                sd1 = np.std(sig1)

                sig1 = (sig1 - mu1) / sd1

                # compare to all other signals with that lead
                for s2 in fields_dict:
                    if s==s2:
                        continue

                    # print(s2, end=' ')
                    if lead in fields_dict[s2]['signame']:
                        # get index of lead in 2nd signal
                        m = [i for i, val in enumerate(fields_dict[s2]['signame']) if val==lead][0]

                        sig2 = (np.copy(signals_dict[s2][:,m])*fields_dict[s2]['gain'][m]).astype(int)

                        # downsample the signal
                        sig2 = downsample_signal(sig2, fields_dict[s2], Fnew=Fnew)

                        # extract the 10 seconds of interest
                        sig2 = sig2[T_START*Fnew:T_END*Fnew]

                        # normalize
                        sig2 = (sig2 - np.mean(sig2)) / np.std(sig2)

                        # run DTW
                        dist, cost, path = mlpy.dtw_std(sig1, sig2, dist_only=False, squared=False)

                        #path[0], sig2[path[1]]
                        #path[0], sig2[path[1]]
                        sig_out = np.array( [path[1], (sig1[path[0]]*sd1)+mu1] ).T

                        np.savetxt('dtw/' + lead + '_' + s + '_to_' + s2 + '.csv',
                                   sig_out, fmt=['%4d','%8.2f'], delimiter=',')
                    else:
                        # the comparison signal does not have the same lead
                        continue
                print() # newline to go to a new signal

            else:
                continue

def normalize_sig(sig):
    return (sig - np.mean(sig)) / np.std(sig)


def sig_distance(sig1, fields1, sig2, fields2, radius, new_fs, max_channels=1, num_secs=10):
    channels_dists = {}
    channels1 = fields1['signame']
    channels2 = fields2['signame']

    common_channels = list(set(channels1).intersection(set(channels2)))
    if len(common_channels) > max_channels:
        common_channels = common_channels[:max_channels]

    start_index = int(DEFAULT_ECG_FS * (ALARM_TIME-num_secs))
    end_index = int(DEFAULT_ECG_FS * ALARM_TIME)

    for channel in common_channels:
        if channel == "RESP":
            continue

        try:
            channel_index1 = channels1.index(channel)
            channel_index2 = channels2.index(channel)

        except Exception as e:
            print(" channels1: {}".format(channels1), end=" ")
            print(" channels2: {}".format(channels2), end=" ")
            print(" common_channels:{}".format(common_channels), end=" ")
            print(e)
            continue

        channel1 = sig1[start_index:end_index,channel_index1]
        channel2 = sig2[start_index:end_index,channel_index2]

        # Downsample
        channel1_sampled = resample(channel1, num_secs*new_fs)
        channel2_sampled = resample(channel2, num_secs*new_fs)

        # Normalize
        channel1_normalized = normalize_sig(channel1_sampled)
        channel2_normalized = normalize_sig(channel2_sampled)

        try:
            if radius > 0:
                distance, path = fastdtw.fastdtw(channel1_normalized, channel2_normalized, radius=radius, dist=euclidean)
            else:
                distance = sum([val**2 for val in (channel1_normalized - channel2_normalized)])

        except Exception as e:
            continue

        channels_dists[channel] = distance

    return channels_dists


def sig_distance_from_file(sig1, fields1, sig2, fields2, new_fs, num_secs=10):
    channels_dists = {}

    sample_name1 = fields1['filename'][0].strip('.mat')
    sample_name2 = fields2['filename'][0].strip('.mat')

    start_index = int(DEFAULT_ECG_FS * (ALARM_TIME-num_secs))
    end_index = int(DEFAULT_ECG_FS * ALARM_TIME)

    pathname = 'dtw_data/*_{}_to_{}.csv'.format(sample_name1, sample_name2)
    matched_filenames = glob.glob(pathname)

    for filename in matched_filenames:
        channel = filename.lstrip('dtw_data/')[:filename.index("_")-1]
        channel_index1 = fields1['signame'].index(channel)
        channel_index2 = fields2['signame'].index(channel)

        channel1 = sig1[start_index:end_index,channel_index1]
        channel2 = sig2[start_index:end_index,channel_index2]

        # Downsample
        channel1_sampled = resample(channel1, num_secs*new_fs)
        channel2_sampled = resample(channel2, num_secs*new_fs)

        with open(filename, 'r') as f:
            reader = csv.DictReader(f, fieldnames=['path0', 'path1'])

            indices = [ [int(row['path0']), int(row['path1'])] for row in reader ]

            channel1_indices = [ index[0] for index in indices ]
            channel2_indices = [ index[1] for index in indices ]

            channel1_warped = [ channel1_sampled[index] for index in channel1_indices ]
            channel2_warped = [ channel2_sampled[index] for index in channel2_indices ]

            # Normalize
            channel1_normalized = normalize_sig(channel1_warped)
            channel2_normalized = normalize_sig(channel2_warped)

            distance = sum([val**2 for val in (channel1_normalized - channel2_normalized)])

            channels_dists[channel] = distance

    return channels_dists


def normalize_distances(channels_dists, normalization='ecg_average', sigtypes=sigtypes_filename):
    if len(channels_dists.keys()) == 0:
        return float('inf')

    if len(channels_dists.keys()) == 1:
        return channels_dists.values().pop()

    ecg_channels = [ channel for channel in channels_dists if get_channel_type(channel, sigtypes) == "ECG" ]
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

        # channels_dists = sig_distance_from_file(test_sig, test_fields, train_sig, train_fields, new_fs)
        # print("sample_name: {}".format(sample_name))
        # print("channels_dists: {}".format(channels_dists))
        # if len(channels_dists.keys()) == 0:
        #     print("Processing sample {} from scratch".format(sample_name))
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
        # print("sample: {}".format(sample_name))
        # print(" predicted: {}".format(predicted))
        # print(" actual: {}".format(actual))
        # print("elapsed: {}".format(datetime.now() - start))

        min_distances[sample_name] = (distance, sample, predicted == actual)

        classification = get_matrix_classification(actual, predicted)
        matrix[classification].append(sample_name)

    return matrix, min_distances


def run(data_path, num_training, arrhythmias, matrix_filename, distances_filename, radius=0, new_fs=DEFAULT_ECG_FS, weighting=1):
    print("Generating sig and fields dicts...")
    sig_dict, fields_dict = read_signals(data_path)
    sig_training, fields_training, sig_testing, fields_testing = \
        get_data(sig_dict, fields_dict, num_training)
    sig_training_by_arrhythmia = { arrhythmia : get_samples_of_type(sig_training, arrhythmia) \
        for arrhythmia in arrhythmias }
    fields_training_by_arrhythmia = { arrhythmia : get_samples_of_type(fields_training, arrhythmia) \
        for arrhythmia in arrhythmias }

    sig_testing_temp = { sample_name : sig_testing[sample_name] for sample_name in sig_testing.keys() if sample_name[0] == 'v' }
    fields_testing_temp = { sample_name : fields_testing[sample_name] for sample_name in fields_testing.keys() if sample_name[0] == 'v' }

    print("Calculating classification accuracy...")
    matrix, min_distances = run_classification( \
        sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing_temp, fields_testing_temp, radius, new_fs, weighting)

    write_json(matrix, matrix_filename)
    write_json(min_distances, distances_filename)



if __name__ == '__main__':
    start = datetime.now()

    new_fs = 125
    num_training = 500
    arrhythmias = ['a', 'b', 't', 'v', 'f']

    run(data_path, num_training, arrhythmias, matrix_filename, distances_filename, radius=0, new_fs=new_fs)

    matrix = read_json(matrix_filename)
    min_distances = read_json(distances_filename)

    counts = { key : len(matrix[key]) for key in matrix.keys() }
    # vtach_counts, vtach_matrix = get_by_arrhythmia(matrix, 'v')

    print("accuracy: {}".format(get_classification_accuracy(matrix)))
    print("score: {}".format(get_score(matrix)))
    print_stats(counts)

    # print("accuracy: {}".format(get_classification_accuracy(vtach_matrix)))
    # print("score: {}".format(get_score(vtach_matrix)))
    # print_stats(vtach_counts)


    # print("\nVTACH STATS")
    # arrhythmia_dict = get_counts_by_arrhythmia(matrix, "v")
    # arrhythmia_counts = { key : arrhythmia_dict[key][0] for key in arrhythmia_dict.keys() }
    # arrhythmia_matrix = { key : arrhythmia_dict[key][1] for key in arrhythmia_dict.keys() }
    # print("accuracy: {}".format(get_classification_accuracy(arrhythmia_matrix)))
    # print("score: {}".format(get_score(arrhythmia_matrix)))
    # print_stats(arrhythmia_counts)
