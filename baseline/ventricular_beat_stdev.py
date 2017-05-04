import sys
sys.path.append("../")
from classifier             import get_baseline, get_power, get_ksqi, get_pursqi
from load_annotations       import *
from fastdtw                import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats            import entropy
from datetime               import datetime
from copy                   import deepcopy
import numpy                as np
import matplotlib.pyplot    as plt
import wfdb
import peakutils
import csv
import os
import json


STD_MULTIPLIER = 1
MIN_DISTANCE_DIFF = 5
# Assuming a max physiologically possible HR of 300
MIN_PEAK_DIST = 60. / 300 * 250
# Assuming a min physiologically possible HR of 30
MAX_PEAK_DIST = 60. / 30 * 250

DEBUG = True

def dprint(*args): 
    if DEBUG: 
        for arg in args: 
            print arg,
        print ""


def is_noisy(
        channel_subsig,
        checks_to_use,
        baseline_threshold=0.75,
        power_threshold=0.9,
        ksqi_threshold=4,
        pursqi_threshold=5
    ): 

    checks = []
    dprint(get_baseline(channel_subsig), get_power(channel_subsig), get_ksqi(channel_subsig))

    # True if passes check
    baseline_check = get_baseline(channel_subsig) > baseline_threshold
    power_check = get_power(channel_subsig) > power_threshold
    ksqi_check = get_ksqi(channel_subsig) > ksqi_threshold
    # pursqi_check = get_pursqi(channel_subsig) > pursqi_threshold
    # checks = [baseline_check, power_check, ksqi_check, pursqi_check]
    
    # TODO: maybe high pass filter instead of using baseline check as a check
    if 'baseline' in checks_to_use: 
        checks.append(baseline_check)

    if 'power' in checks_to_use: 
        checks.append(power_check)

    if 'ksqi' in checks_to_use: 
        checks.append(ksqi_check)

    return not all(checks)


def get_adjusted_ann_indices(annotation, ann_index, start_ratio=1/3.): 
    a = annotation[ann_index-1]
    b = annotation[ann_index]
    c = annotation[ann_index+1]

    end_ratio = 1-start_ratio

    ann_start_index = b - start_ratio*(b-a)
    ann_end_index = b + end_ratio*(c-b)

    return ann_start_index, ann_end_index


## 
#  Returns self_beats, a list of: 
#       annotation index
#       beat_sig
#  for regular beats detected in own patient's signal
##
def get_self_beats(
        channel_sig, 
        annotation, 
        sample_name, 
        checks_to_use=['baseline', 'power', 'ksqi'],
        num_self_beats=20,
        window_increment=10, 
        fs=250.): 

    self_beats = []

    # Get self beats in first 2 min
    for start_time in range(0, 120-window_increment+1, window_increment): 
        end_time = start_time + window_increment
        start_index = int(start_time * fs)
        end_index = int(end_time * fs)

        channel_subsig = channel_sig[start_index:end_index]
        # print start_index, end_index,

        if not is_noisy(channel_subsig, checks_to_use): 
            for ann_index in range(1, len(annotation)-1):
                # TODO: update to have the start and end index be smoothed over past values
                ann_start_index, ann_end_index = get_adjusted_ann_indices(annotation, ann_index)

                # If beat annotation in clean (not noisy) data range
                if ann_start_index > start_index and ann_end_index < end_index: 
                    beat_sig = channel_sig[int(ann_start_index):int(ann_end_index)]

                    peaks = peakutils.indexes(beat_sig, thres=0.75*max(beat_sig), min_dist=MIN_PEAK_DIST)

                    # if DEBUG: 
                    #     plt.figure()
                    #     plt.plot(peaks, [beat_sig[index] for index in peaks], 'ro')
                    #     plt.plot(beat_sig)
                    #     plt.show()

                    if len(peaks) < 3: 
                        self_beats.append((annotation[ann_index], beat_sig))

                if len(self_beats) >= num_self_beats: 
                    break

    dprint("Found", len(self_beats), "self beats.")

    if DEBUG: 
        plt.figure()
        for i, beat in enumerate(self_beats):
            plt.subplot(5, 4, i+1)
            plt.plot(beat[1])
        plt.show()

    return self_beats


def get_best_self_beats(channel_sig, full_annotation, sample_name): 
    self_beats = get_self_beats(channel_sig, full_annotation, sample_name)

    if len(self_beats) == 0: 
        self_beats = get_self_beats(channel_sig, full_annotation, sample_name, ['power', 'ksqi'])

    if len(self_beats) == 0: 
        self_beats = get_self_beats(channel_sig, full_annotation, sample_name, ['power'])

    if len(self_beats) == 0: 
        dprint("No self beats found for", sample_name)

    return self_beats


def normalize_sig(sig): 
    return (sig - np.mean(sig)) / np.std(sig)

# ##
# #  Returns mean and stdev based on new normal beats found in the following 2 min
# ##
# def get_baseline_metrics_normal(
#         channel_sig, 
#         self_beats, 
#         annotation, 
#         window_start=120,
#         window_end=240,
#         window_increment=10,
#         fs=250.): 

#     min_distances = []

#     # Get self beats in first 2 min
#     # TODO: change this? 
#     for start_time in range(window_start, window_end-window_increment, window_increment): 
#         end_time = start_time + window_increment
#         start_index = int(start_time * fs)
#         end_index = int(end_time * fs)

#         channel_subsig = channel_sig[start_index:end_index]

#         if is_noisy(channel_subsig): 
#             continue

#         for ann_index in range(1, len(annotation)-1):
#             # TODO: update to have the start and end index be smoothed over past values
#             ann_start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
#             ann_end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2

#             # If beat annotation in clean (not noisy) data range
#             if ann_start_index > start_index and ann_end_index < end_index: 
#                 beat_sig = channel_sig[ann_start_index:ann_end_index]
                
#                 distances = get_dtw_distances(beat_sig, self_beats)
#                 min_distances.append(min(distances))

#                 if len(min_distances) >= 20: 
#                     return np.mean(min_distances), np.std(min_distances)

#     return -1, -1


##
#  Returns mean and stdev comparing against every other self beat in bank
##
def get_baseline_distances(self_beats, radius=250): 

    # if DEBUG: 
    #     plt.figure()
    #     for i, beat in enumerate(self_beats):
    #         plt.subplot(5, 4, i+1)
    #         plt.plot(beat[1])
    #     plt.show()

    # Pairwise compare with every other self beat
    all_distances = [] 

    for i in range(len(self_beats)): 
        distances = []

        for j in range(len(self_beats)): 
            if i != j:
                i_beat = self_beats[i][1]
                j_beat = self_beats[j][1]

                distance, path = fastdtw(normalize_sig(i_beat), normalize_sig(j_beat), radius=radius, dist=euclidean)
                distances.append(distance)

        all_distances.append(distances)

    return all_distances

def get_kl_dist(distances): 
    return [ val if val > 0 else 0.000001 for val in np.histogram(distances, bins=2000)[0] ]


def get_baseline_metrics(metric, baseline_distances): 
    top_level_distances = []

    if metric == 'kl': 
        flat_distances = [ item for sublist in baseline_distances for item in sublist ]
        flat_hist = get_kl_dist(flat_distances)

        for sublist in baseline_distances: 
            sublist_hist = get_kl_dist(sublist)
            kl_distance = entropy(sublist_hist, flat_hist)
            top_level_distances.append(kl_distance)

    elif metric == 'min': 
        top_level_distances = [ min(sublist) for sublist in baseline_distances ]

    elif metric == 'mean': 
        top_level_distances = [ np.mean(sublist) for sublist in baseline_distances ]

    else: 
        raise Exception("Unrecognized metric: ", metric)

    metric_info = [ np.mean(top_level_distances), np.std(top_level_distances) ]
    if metric == 'kl': 
        metric_info.append(deepcopy(baseline_distances))

    return metric_info


def get_dtw_distances(beat_sig, self_beats, radius=250): 
    distances = []
    beat_sig_normalized = normalize_sig(beat_sig)

    figure_num = 1

    for self_beat in self_beats: 
        self_beat_normalized = normalize_sig(self_beat[1])

        try: 
            distance, path = fastdtw(beat_sig_normalized, self_beat_normalized, radius=radius, dist=euclidean)
            distances.append(distance)

            plt.subplot(5, 4, figure_num)
            plt.title(str(int(distance)))
            plt.plot(self_beat_normalized, 'b-')
            plt.plot(beat_sig_normalized, 'r-')
            plt.axis('off')
            figure_num += 1

        except Exception as e: 
            print e

    plt.show()
    return distances


##
#   Determine if ventricular beat is stdev or not
#       metric: string indicating metric ('kl', 'min', 'mean')
#       metric info: list of relevant metric info
#           if 'kl': [ mean, std, baseline_distances ]
#           else: [ mean, std ]
##
def is_ventricular_beat_stdev(beat_sig, self_beats, metric, metric_info, threshold): 
    plt.figure(figsize=[12, 8])
    plt.title(str(metric_info[0]) + " " + str(metric_info[1]))
    beat_distances = get_dtw_distances(beat_sig, self_beats)

    if len(beat_distances) == 0: 
        # TODO: maybe return false because probably contains inf/nan which is invalid data
        return True

    if metric == 'kl': 
        baseline_distances = metric_info[2]
        flat_distances = [ item for sublist in baseline_distances for item in sublist ]
        flat_hist = get_kl_dist(flat_distances)
        beat_hist = get_kl_dist(beat_distances)

        metric_distance = entropy(beat_hist, flat_hist)
        
    elif metric == "min": 
        metric_distance = min(beat_distances)

    elif metric == 'mean': 
        metric_distance = np.mean(beat_distances)

    else: 
        raise Exception("Unrecognized metric type: ", metric)

    dprint("distance: ", metric_distance, metric_distance > threshold)

    if metric_distance > threshold: 
        return True

    return False


##
# beats is a list of tuples containing: 
#       annotation of beat QRS
#       start and end indices
#       sig of beat
##
def get_ventricular_beats(beats, self_beats, metric, metric_info):
    ventricular_beats = []
    nonventricular_beats = []

    mean = metric_info[0]
    std = metric_info[1]

    # TODO: optimize hyperparameter STD_MULTIPLIER and MIN_DISTANCE_DIFF
    threshold = max(mean + std * STD_MULTIPLIER, mean + MIN_DISTANCE_DIFF)
    dprint("mean: ", metric_info[0], "std: ", metric_info[1], "threshold: ", threshold)

    for beat in beats: 
        beat_sig = beat[1]

        if is_ventricular_beat_stdev(beat_sig, self_beats, metric, metric_info, threshold): 
            ventricular_beats.append(beat)
        else: 
            nonventricular_beats.append(beat)

    return ventricular_beats, nonventricular_beats


##
# Returns beats (list of tuples): 
#       annotation of beat QRS
#       start and end indices
#       sig of beat
##
def get_alarm_beats(channel_sig, annotation): 
    beats = []
    for ann_index in range(1, len(annotation)-1):
        # Assumes a beat starts start_ratio (default 1/3) before the annotation 
        # and ends end_ratio (default 2/3) after annotation
        # TODO: update this to update dynamically based on past values
        start_index, end_index = get_adjusted_ann_indices(annotation, ann_index)

        indices = (start_index, end_index)
        beat_sig = channel_sig[int(indices[0]):int(indices[1])]
        beat = (annotation[ann_index], beat_sig)

        if len(beat_sig) > MIN_PEAK_DIST and len(beat_sig) < MAX_PEAK_DIST: 
            beats.append(beat)

    return beats


##
#   Plot histogram of all pairwise distances between self beatst
##
def plot_metrics(baseline_distances, metric, metric_info): 
    flat_distances = [ item for sublist in baseline_distances for item in sublist ]
    mean = metric_info[0]
    std = metric_info[1]
    multipliers = [0.5, 1, 2, 3, 4, 5]

    # Plot all flat distances with mean + std + various multipliers
    plt.figure()
    plt.hist(flat_distances, edgecolor='black')
    plt.axvline(mean, color='r')
    for multiplier in multipliers: 
        plt.axvline(x=mean + std*multiplier, color='g')

    plt.show()


    # Plot individual distance distributions against flat distances
    plt.figure(figsize=[12, 8])
    for index, distances in enumerate(baseline_distances): 
        plt.subplot(5, 4, index+1)
        plt.hist(flat_distances, color='blue', edgecolor='black')
        plt.hist(distances, color='red', edgecolor='black')
        if metric == 'min': 
            plt.axvline(x=min(distances), color='r')
        elif metric == 'mean': 
            plt.axvline(x=np.mean(distances), color='r')

    plt.show()


def plot_self_beat_comparison(self_beats): 
    for i in range(len(self_beats)): 
        plt.figure(figsize=[12, 8])
        figure_num = 1

        for j in range(len(self_beats)): 
            if i != j:
                i_beat = self_beats[i][1]
                j_beat = self_beats[j][1]

                plt.subplot(5, 4, figure_num)
                plt.plot(normalize_sig(i_beat), 'b-')
                plt.plot(normalize_sig(j_beat), 'r-')
                plt.axis('off')
                figure_num += 1

    plt.show()


def filter_out_nan(beats): 
    filtered = []

    for beat in beats: 
        beat_sig = beat[1]
        if not np.isnan(np.sum(beat_sig)): 
            filtered.append(beat)

    return filtered


def ventricular_beat_annotations_dtw(
        channel_sig, 
        ann_path, 
        sample_name,    
        metric,
        start_time, 
        end_time,
        ann_type,
        force=False,
        file_prefix="../sample_data/vtach_beat_ann/std/baseline_distances_",
        ann_fs=250.):

    baseline_dist_filename = file_prefix + sample_name + ".json"

    dprint("Finding alarm beats...")
    annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, start_time, end_time).annsamp
    alarm_beats = get_alarm_beats(channel_sig, annotation)

    dprint("Finding self beats...")
    # Full annotation except for when the alarm signal starts (usually last 10 seconds)
    full_annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, 0, start_time).annsamp
    self_beats = get_best_self_beats(channel_sig, full_annotation, sample_name)

    if os.path.isfile(baseline_dist_filename) and not force:
        dprint("Loading baseline distances from file...")
        with open(baseline_dist_filename, 'r') as f: 
            baseline_distances = json.load(f)
    else:  
        dprint("Calculating baseline distances...")
        baseline_distances = get_baseline_distances(self_beats)
        
        dprint("Writing baseline distances to file...")
        with open(baseline_dist_filename, 'w') as f: 
            json.dump(baseline_distances, f)

    try: 
        dprint("Calculating baseline metrics...")
        metric_info = get_baseline_metrics(metric, baseline_distances)
    except Exception as e: 
        print "sample_name: ", sample_name, e
        return [], []

    # plot_metrics(baseline_distances, metric, metric_info)
    # plot_self_beat_comparison(self_beats)

    dprint("Classifying alarm beats...")
    ventricular_beats, nonventricular_beats = get_ventricular_beats(alarm_beats, self_beats, metric, metric_info)
    print "ventricular: ", len(ventricular_beats), ventricular_beats
    vtach_beats = filter_out_nan(ventricular_beats)
    print "filtered: ", len(vtach_beats), vtach_beats

    # Only find distances if ventricular beats were found
    if len(vtach_beats) > 1: 
        ventricular_distances = get_baseline_distances(vtach_beats)
        ventricular_mean, ventricular_std = get_baseline_metrics('min', ventricular_distances)

        # If ventricular beats don't look very similar, mark as noise instead
        if ventricular_mean > 20 and ventricular_std > 15 and ventricular_mean > ventricular_std: 
            vtach_beats = []

    ventricular_beat_anns = [ beat[0] for beat in vtach_beats ]
    nonventricular_beat_anns = [ beat[0] for beat in nonventricular_beats ]

    return ventricular_beat_anns, nonventricular_beat_anns



def write_vtach_beats_files(
        data_path, 
        ann_path, 
        output_path, 
        ecg_ann_type, 
        start_time, 
        end_time, 
        metric): 

    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)

            if sample_name[0] != 'v':
                continue

            sig, fields = wfdb.srdsamp(data_path + sample_name)
            if "II" not in fields['signame']: 
                print "Lead II not found for sample: ", sample_name
                continue

            output_filename = output_path + sample_name + "_fp_" + metric + ".csv"

            if os.path.isfile(output_filename): 
                continue

            print sample_name

            channel_index = fields['signame'].index("II")
            ann_type = ecg_ann_type + str(channel_index)

            start = datetime.now()

            with open(output_filename, "w") as f:
                channel_sig = sig[:,channel_index]

                vtach, nonvtach = ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, metric, start_time, end_time, ann_type)

                writer = csv.writer(f)
                writer.writerow(['ann_index', 'is_true_beat'])

                for beat in vtach: 
                    writer.writerow([beat, 1])
                for beat in nonvtach: 
                    writer.writerow([beat, 0])

            print "sample_name: ", sample_name, " elapsed: ", datetime.now() - start

def run_one_sample():
    # sample_name = "v100s" # false alarm
    # sample_name = "v141l" # noisy at beginning
    # sample_name = "v159l" # quite clean
    # sample_name = "v206s" # high baseline
    # sample_name = "v143l"
    # sample_name = "v696s"
    sample_name = "v797l"
    channel_index = 0
    ann_fs = 250.
    ann_type = 'gqrs' + str(channel_index)

    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channel_sig = sig[:,channel_index]

    vtach_beats, nonvtach_beats = ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, 'kl', start_time, end_time, ann_type)

    plt.figure(figsize=[8,5])
    plt.plot(channel_sig[int(start_time*250.):int(end_time*250.)],'b-')
    plt.plot([ int(index-250.*start_time) for index in nonvtach_beats ], [channel_sig[int(index)] for index in nonvtach_beats], 'bo', markersize=8)
    plt.plot([ int(index-250.*start_time) for index in vtach_beats ], [ channel_sig[int(index)] for index in vtach_beats ], 'ro', markersize=8)
    plt.show()


data_path = "../sample_data/challenge_training_data/"
ann_path = "../sample_data/challenge_training_multiann/"
fp_path = "../sample_data/fplesinger_data/"
output_path = "../sample_data/vtach_beat_ann/std/"
start_time = 296
end_time = 300
ecg_ann_type = "gqrs"

run_one_sample()

# write_vtach_beats_files(data_path, ann_path, output_path, ecg_ann_type, start_time, end_time, 'min')


# sig, fields = wfdb.rdsamp(data_path + sample_name)
# channel_sig = sig[:,channel_index]

# annotation = wfdb.rdann(ann_path + sample_name, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs).annsamp
# print annotation

# beats = get_beats(channel_sig, annotation)


# for beat in beats: 
#   indices = beat[0]
#   beat_sig = beat[1]
#   time_vector = np.linspace(indices[0], indices[1], len(beat_sig))

#   whole_sig = channel_sig[250*start:250*end]
#   sig_time_vector = np.linspace(250*start, 250*end, len(whole_sig))

#   annotation_y = [ channel_sig[ann_t] for ann_t in annotation ]

#   plt.figure()
#   plt.plot(sig_time_vector, whole_sig, 'b')
#   plt.plot(time_vector, beat_sig, 'r')
#   plt.plot(annotation, annotation_y, 'go')
#   plt.show()



# print ""
# print annotation[0] / float(250.)