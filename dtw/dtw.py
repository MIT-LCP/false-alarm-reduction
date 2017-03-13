from utils           import *
from parameters      import *
from datetime        import datetime
import numpy         as np
import sklearn
import fastdtw
import wfdb
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


def channel_distance(timeseries1, timeseries2, dist_metric=abs_value_dist_metric):
    ts1 = np.array(timeseries1)
    ts2 = np.array(timeseries2)

    ts1_size = len(timeseries1)
    ts2_size = len(timeseries2)

    cost = float("inf") * np.ones((ts1_size, ts2_size))

    cost[0,0] = dist_metric(ts1[0], ts2[0])

    for i in range(1,ts1_size):
        cost[i,0] = cost[i-1,0] + dist_metric(ts1[i], ts2[0])

    for j in range(1,ts2_size):
        cost[0,j] = cost[0,j-1] + dist_metric(ts1[0], ts2[j])

    for i in range(1,ts1_size):
        for j in range(1,ts2_size):
            min_prev_cost = min(cost[i,j-1], cost[i-1,j], cost[i-1,j-1])
            cost[i,j] = min_prev_cost + dist_metric(ts1[i], ts2[j])

    return cost[-1, -1]


def sig_distance(sig1, fields1, sig2, fields2, max_channels=1):
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

        channels_dists[channel] = channel_distance(channel1, channel2)

    return channels_dists


def normalize_distances(channels_dists, normalization='ecg_average', sigtypes_filename='../sample_data/sigtypes'):
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

    arrhythmia = get_arrhythmia_type(test_fields)
    sig_training = sig_training_by_arrhythmia[arrhythmia]
    fields_training = fields_training_by_arrhythmia[arrhythmia]

    for sample_name, train_sig in sig_training.items():
        train_fields = fields_training[sample_name]
        channels_dists = sig_distance(test_sig, test_fields, train_sig, train_fields)
        distance = normalize_distances(channels_dists)

        print "training sample:", sample_name, " distance:", distance, " min_distance:", min_distance

        if distance < min_distance:
            min_distance = distance
            min_label = is_true_alarm(train_fields)

    return min_label

## Get classification accuracy of testing based on training set
# sig_training_by_arrhythmia
def get_classification_accuracy(sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing):
    num_correct = 0

    for sample_name, test_sig in sig_testing.items():
        test_fields = fields_testing[sample_name]

        print "sample: ", sample_name

        prediction = predict(test_sig, test_fields, sig_training_by_arrhythmia, fields_training_by_arrhythmia)
        actual = is_true_alarm(test_fields)
        print "prediction:", prediction, " actual:", actual

        if prediction == actual:
            num_correct += 1

    return float(num_correct) / len(sig_testing)


if __name__ == '__main__':
    num_training = 500
    arrhythmias = ['a', 'b', 't', 'v', 'f']

    start = datetime.now()
    print "start: ", start

    print "Generating sig and fields dicts..."
    sig_dict, fields_dict = read_signals(data_path)
    sig_training, fields_training, sig_testing, fields_testing = \
        get_data(sig_dict, fields_dict, num_training)
    sig_training_by_arrhythmia = { arrhythmia : get_samples_of_type(sig_training, arrhythmia) \
        for arrhythmia in arrhythmias }
    fields_training_by_arrhythmia = { arrhythmia : get_samples_of_type(fields_training, arrhythmia) \
        for arrhythmia in arrhythmias }

    print "Calculating classification accuracy..."
    print get_classification_accuracy(sig_training_by_arrhythmia, fields_training_by_arrhythmia, sig_testing, fields_testing)

    print datetime.now() - start