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

def generate_training(filename): 
    training = []

    with open(filename, 'r') as f: 
        reader = csv.DictReader(f)

        for row in reader: 
            sample_name = row['sample_name']
            is_true_beat = int(row['is_vtach']) == 1

            start_time = float(row['start_time'])
            end_time = float(row['end_time'])

            # peak_time = float(row['peak_time'])
            # start_time = peak_time - AVERAGE_START_DIFF
            # end_time = peak_time + AVERAGE_END_DIFF

            sig, fields = wfdb.rdsamp(data_path + sample_name)
            start_index = int(start_time*250.)
            end_index = int(end_time*250.)
            channel_index = fields['signame'].index(row['lead'])

            beat_sig = sig[start_index:end_index,channel_index]

            training.append((beat_sig, is_true_beat, sample_name))

    return training

def get_self_beats(channel_sig, annotation, sample_name): 
    ### TODO: add quality check before adding training beats

    training_beats = []

    for ann_index in range(1, len(annotation)-1):
        start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
        end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2
        beat_sig = channel_sig[start_index:end_index]

        training_beats.append((beat_sig, False, sample_name))

        if len(training_beats) >= 5: 
            break

    return training_beats


def normalize_sig(sig): 
    return (sig - np.mean(sig)) / np.std(sig)

def is_ventricular_beat(beat_sig, training_beats): 
    # Euclidean distance between beat sig and a flat line to represent noise --> not a ventricular beat
    min_distance = sum([val**2 for val in normalize_sig(beat_sig)])
    classification = False
    min_sample_name = ""
    min_training_beat = []

    # figure_num = 1
    # plt.figure(figsize=[12, 12])

    for beat_tuple in training_beats:
        training_beat = beat_tuple[0]
        is_true_beat = beat_tuple[1]
        sample_name = beat_tuple[2]

        training_beat_normalized = normalize_sig(training_beat)
        beat_sig_normalized = normalize_sig(beat_sig)

        # if len(training_beat) > len(beat_sig): 
        #     training_beat_normalized = normalize_sig(training_beat[:len(beat_sig)])

        # elif len(beat_sig) > len(training_beat): 
        #     beat_sig_normalized = normalize_sig(beat_sig[:len(training_beat)])
        # distance = sum([val**2 for val in (training_beat_normalized - beat_sig_normalized)])

        try: 
            distance, path = fastdtw(beat_sig_normalized, training_beat_normalized, radius=250, dist=euclidean)
        except Exception as e: 
            distance = float('inf')
            print "Error with training sample: ", sample_name, e

        # print sample_name, distance, is_true_beat
        # plt.subplot(9, 5, figure_num)
        # plt.title(str(int(distance)) + " " + str(is_true_beat))
        # plt.plot(training_beat_normalized, 'b-')
        # plt.axis('off')
        # plt.plot(beat_sig_normalized, 'r-')
        # figure_num += 1

        if distance < min_distance: 
            min_distance = distance
            classification = is_true_beat
            min_sample_name = sample_name
            min_training_beat = training_beat_normalized

    # print("min: ", min_sample_name, min_distance, classification)
    # plt.show()

    # if classification: 
    #     plt.figure()
    #     plt.plot(min_training_beat, 'b-')
    #     plt.plot(beat_sig_normalized, 'r-')
    #     plt.title(min_sample_name + " " + str(int(min_distance)) + " " + str(classification))
    #     plt.show()

    return classification


def get_ventricular_beats(beats, training_beats):
    ventricular_beats = []
    nonventricular_beats = []

    for beat in beats: 
        beat_sig = beat[2]
        print "annotation index: ", beat[0]

        if is_ventricular_beat(beat_sig, training_beats): 
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
def get_beats(channel_sig, annotation): 
    beats = []
    for ann_index in range(1, len(annotation)-1):
        # Assumes a beat starts halfway between annotations and ends halfway between annotations
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

def ventricular_beat_annotations_dtw(
        channel_sig, 
        ann_path, 
        sample_name, 
        start_time, 
        end_time,
        ann_type,
        ann_fs=250.,
        training_filename="../sample_data/vtach_beats.csv"):

    training_beats = generate_training(training_filename)

    annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, start_time, end_time)[0]
    full_annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, 0, start_time)[0]
    sample_training_beats = get_self_beats(channel_sig, full_annotation, sample_name)

    beats = get_beats(channel_sig, annotation)
    ventricular_beats, nonventricular_beats = get_ventricular_beats(beats, sample_training_beats + training_beats)
    # ventricular_beats, nonventricular_beats = get_ventricular_beats(beats, training_beats)

    ventricular_beat_annotations = [ beat[0] for beat in ventricular_beats ]
    nonventricular_beat_annotations = [ beat[0] for beat in nonventricular_beats ]

    return ventricular_beat_annotations, nonventricular_beat_annotations


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




training_filename = "../sample_data/vtach_beats.csv"
data_path = "../sample_data/challenge_training_data/"
ann_path = "../sample_data/challenge_training_multiann/"
output_path = "../sample_data/vtach_beat_ann/"

# sample_name = "v127l"
sample_name = "v141l"
ecg_ann_type = 'gqrs'
start_time = 296
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