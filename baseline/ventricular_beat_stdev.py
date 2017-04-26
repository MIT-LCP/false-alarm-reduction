import sys
sys.path.append("../")
from classifier             import get_baseline, get_power, get_ksqi, get_pursqi
from load_annotations       import *
from fastdtw                import fastdtw
from scipy.spatial.distance import euclidean
from datetime               import datetime
import numpy                as np
import matplotlib.pyplot    as plt
import wfdb
import csv
import os



AVERAGE_START_DIFF = 0.25
AVERAGE_END_DIFF = 0.35

headers = [
    'num', 
    'sample_name', 
    'arrhythmia', 
    'is_true_beat', 
    'start_time', 
    'end_time',
    'comments'
]

def is_noisy(
        channel_subsig,
        baseline_threshold=0.9,
        power_threshold=0.9,
        ksqi_threshold=5,
        pursqi_threshold=5
    ): 

    # True if passes check
    baseline_check = get_baseline(channel_subsig) > baseline_threshold
    power_check = get_power(channel_subsig) > power_threshold
    ksqi_check = get_ksqi(channel_subsig) > ksqi_threshold
    # pursqi_check = get_pursqi(channel_subsig) > pursqi_threshold
    # checks = [baseline_check, power_check, ksqi_check, pursqi_check]
    checks = [baseline_check, power_check, ksqi_check]

    return not all(checks)


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
        num_self_beats=20,
        window_increment=10, 
        fs=250.): 

    self_beats = []

    # Get self beats in first 2 min
    # TODO: change this? 
    for start_time in range(0, 120-window_increment, window_increment): 
        end_time = start_time + window_increment
        start_index = int(start_time * fs)
        end_index = int(end_time * fs)

        channel_subsig = channel_sig[start_index:end_index]

        if not is_noisy(channel_subsig): 
            for ann_index in range(1, len(annotation)-1):
                # TODO: update to have the start and end index be smoothed over past values
                ann_start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
                ann_end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2

                # If beat annotation in clean (not noisy) data range
                if ann_start_index > start_index and ann_end_index < end_index: 
                    beat_sig = channel_sig[ann_start_index:ann_end_index]
                    self_beats.append((annotation[ann_index], beat_sig))

                if len(self_beats) >= num_self_beats: 
                    return self_beats

    print "Found", len(self_beats), "self beats"
    return self_beats


def normalize_sig(sig): 
    return (sig - np.mean(sig)) / np.std(sig)

##
#  Returns mean and stdev 
##
def get_baseline_metrics(
        channel_sig, 
        self_beats, 
        annotation, 
        window_start=120,
        window_end=240,
        window_increment=10,
        fs=250.): 

    min_distances = []

    # Get self beats in first 2 min
    # TODO: change this? 
    for start_time in range(window_start, window_end-window_increment, window_increment): 
        end_time = start_time + window_increment
        start_index = int(start_time * fs)
        end_index = int(end_time * fs)

        channel_subsig = channel_sig[start_index:end_index]

        if is_noisy(channel_subsig): 
            continue

        for ann_index in range(1, len(annotation)-1):
            # TODO: update to have the start and end index be smoothed over past values
            ann_start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
            ann_end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2

            # If beat annotation in clean (not noisy) data range
            if ann_start_index > start_index and ann_end_index < end_index: 
                beat_sig = channel_sig[ann_start_index:ann_end_index]
                
                distances = get_dtw_distances(beat_sig, self_beats)
                min_distances.append(min(distances))

                if len(min_distances) >= 20: 
                    return np.mean(min_distances), np.std(min_distances)

    return -1, -1



def get_dtw_distances(beat_sig, self_beats): 
    distances = []
    beat_sig_normalized = normalize_sig(beat_sig)

    for self_beat in self_beats: 

        self_beat_normalized = normalize_sig(self_beat[1])

        try: 
            distance, path = fastdtw(beat_sig_normalized, self_beat_normalized, radius=250, dist=euclidean)
            distances.append(distance)
        except Exception as e: 
            print "Error with training sample: ", sample_name, e

    return distances



def is_ventricular_beat_stdev(beat_sig, self_beats, mean, stdev): 
    distances = get_dtw_distances(beat_sig, self_beats)

    if len(distances) > 0: 
        min_distance = min(distances)
        print "min_distance: ", min_distance

        if abs(min_distance - mean) > stdev: 
            return True

        return False

    # TODO: maybe return false because probably contains inf/nan which is invalid data
    return True



##
# beats is a list of tuples containing: 
#       annotation of beat QRS
#       start and end indices
#       sig of beat
##
def get_ventricular_beats(beats, self_beats, mean, stdev):
    ventricular_beats = []
    nonventricular_beats = []

    for beat in beats: 
        beat_sig = beat[2]
        # print "annotation index: ", beat[0]

        if is_ventricular_beat_stdev(beat_sig, self_beats, mean, stdev): 
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
        # Assumes a beat starts halfway between annotations and ends halfway between annotations
        # TODO: update this to update dynamically based on past values
        start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
        end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2

        # Assumes a beat starts 250ms before the annotation and ends 250 ms after the annotation
        # start_index = annotation[ann_index] - int(AVERAGE_START_DIFF * 250.)
        # end_index = annotation[ann_index] + int(AVERAGE_END_DIFF * 250.) + 1

        indices = (start_index, end_index)
        beat_sig = channel_sig[indices[0]:indices[1]]
        beat = (annotation[ann_index], indices, beat_sig)

        beats.append(beat)

    return beats

def plot_metrics(self_beats): 
    distances = [] 

    for i in range(len(self_beats)): 
        for j in range(len(self_beats)): 
            if i != j: 
                i_beat = self_beats[i][1]
                j_beat = self_beats[j][1]

                distance, path = fastdtw(normalize_sig(i_beat), normalize_sig(j_beat), radius=250, dist=euclidean)

                distances.append(distance)

    print distances

    plt.figure()
    plt.hist(distances)
    plt.show()


def ventricular_beat_annotations_dtw(
        channel_sig, 
        ann_path, 
        sample_name, 
        start_time, 
        end_time,
        ann_type,
        ann_fs=250.):

    annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, start_time, end_time)[0]
    alarm_beats = get_alarm_beats(channel_sig, annotation)

    # Full annotation except for when the alarm signal starts (usually last 10 seconds)
    full_annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, 0, start_time)[0]
    self_beats = get_self_beats(channel_sig, full_annotation, sample_name)

    plot_metrics(self_beats)

    mean, stdev = get_baseline_metrics(channel_sig, self_beats, full_annotation)
    print mean, stdev

    ventricular_beats, nonventricular_beats = get_ventricular_beats(alarm_beats, self_beats, mean, stdev)

    ventricular_beat_anns = [ beat[0] for beat in ventricular_beats ]
    nonventricular_beat_anns = [ beat[0] for beat in nonventricular_beats ]

    return ventricular_beat_anns, nonventricular_beat_anns


def write_vtach_beats_files(data_path, ann_path, output_path, ecg_ann_type, start_time, end_time): 
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)

            if sample_name[0] != 'v':
                continue

            sig, fields = wfdb.rdsamp(data_path + sample_name)
            if "II" not in fields['signame']: 
                print "Lead II not found for sample: ", sample_name
                continue

            output_filename = output_path + sample_name + ".csv"

            if os.path.isfile(output_filename): 
                continue

            channel_index = fields['signame'].index("II")
            ann_type = ecg_ann_type + str(channel_index)

            start = datetime.now()

            with open(output_filename, "w") as f:
                channel_sig = sig[:,channel_index]

                vtach_beats, nonvtach_beats = ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, start_time, end_time, ann_type)

                writer = csv.writer(f)
                writer.writerow(['ann_index', 'is_true_beat'])

                for beat in vtach_beats: 
                    writer.writerow([beat, 1])
                for beat in nonvtach_beats: 
                    writer.writerow([beat, 0])

            print "sample_name: ", sample_name, " elapsed: ", datetime.now() - start




data_path = "../sample_data/challenge_training_data/"
ann_path = "../sample_data/challenge_training_multiann/"
output_path = "../sample_data/vtach_beat_ann/"

# sample_name = "v127l"
# sample_name = "v141l" # noisy at beginning
sample_name = "v159l" # quite clean
ecg_ann_type = 'gqrs'
start_time = 290
end_time = 300
channel_index = 0
ann_fs = 250.
ann_type = 'gqrs' + str(channel_index)

sig, fields = wfdb.rdsamp(data_path + sample_name)
channel_sig = sig[:,channel_index]

vtach_beats, nonvtach_beats = ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, start_time, end_time, ann_type)
vtach_indices = [ ann - start_time * 250. for ann in vtach_beats ]
nonvtach_indices = [ ann - start_time * 250. for ann in nonvtach_beats ]

plt.figure(figsize=[8,5])
plt.plot(channel_sig[int(start_time*250.):int(end_time*250.)],'b-')
plt.plot(nonvtach_indices, [channel_sig[int(index)] for index in nonvtach_indices], 'bo', markersize=8)
plt.plot(vtach_indices, [ channel_sig[int(index)] for index in vtach_indices ], 'ro', markersize=8)
plt.show()

start_time = 290
end_time = 300
ecg_ann_type = "gqrs"

# write_vtach_beats_files(data_path, ann_path, output_path, ecg_ann_type, start_time, end_time)


# sig, fields = wfdb.rdsamp(data_path + sample_name)
# channel_sig = sig[:,channel_index]

# annotation = wfdb.rdann(ann_path + sample_name, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)[0]
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