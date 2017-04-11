from load_annotations 		import *
import numpy 				as np
import matplotlib.pyplot    as plt
import wfdb
import csv



data_path = "../sample_data/challenge_training_data/"
ann_path = "../sample_data/challenge_training_multiann/"

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
    training = {}

    with open(filename, 'r') as f: 
        reader = csv.DictReader(f, fieldnames=headers)

        for row in reader: 
            sample_name = row['sample_name']
            is_true_beat = row['is_true_beat'] == '1'

            if is_true_beat: 
                start_time = float(row['start_time'])
                end_time = float(row['end_time'])

                sig, fields = wfdb.rdsamp(data_path + sample_name)
                start_index = int((290+start_time)*250.)
                end_index = int((290+end_time)*250.)

                # TODO: double check channel 
                beat_sig = sig[start_index:end_index,0]

                training[sample_name] = (beat_sig, is_true_beat)

            else: 
                training[sample_name] = ([], is_true_beat)

    return training




def is_ventricular_beat(beat_sig): 
    min_distance = float('inf')
    classification = False

    for sample_name in training_beats.keys():
        training_beat = training_beats[sample_name][0]

        distance = sum([val**2 for val in (training_beat - beat_sig)])

        if distance < min_distance: 
            min_distance = distance
            classification = training_beats[sample_name][1]

    return classification

def get_ventricular_beats(beats):
	ventricular_beats = []
	nonventricular_beats = []

	for beat in beats: 
		beat_sig = beat[2]

		if is_ventricular_beat(beat_sig): 
			ventricular_beats.append(beat)
		else: 
			nonventricular_beats.append(beat)

	return ventricular_beats, nonventricular_beats


def get_beats(channel_sig, annotation): 
	beats = []
	for ann_index in range(1, len(annotation)-1):
		start_index = (annotation[ann_index-1] + annotation[ann_index]) / 2
		end_index = (annotation[ann_index] + annotation[ann_index+1]) / 2

		indices = (start_index, end_index)
		beat_sig = channel_sig[indices[0]:indices[1]]
		beat = (annotation[ann_index], indices, beat_sig)

		beats.append(beat)

	return beats

def ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, start_time, end_time, ann_type): 
    annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, start_time, end_time)[0]

    beats = get_beats(channel_sig, annotation)
    ventricular_beats, nonventricular_beats = get_ventricular_beats(beats)

    ventricular_beat_annotations = [ beat[0] for beat in ventricular_beats ]

    return ventricular_beat_annotations


sample_name = "v482s"
ecg_ann_type = 'gqrs'
start_time = 290
end_time = 300
channel_index = 0
ann_fs = 250.
ann_type = 'gqrs' + str(channel_index)

sig, fields = wfdb.rdsamp(data_path + sample_name)
channel_sig = sig[:,channel_index]

# ventricular_beat_annotations_dtw(channel_sig, ann_path, sample_name, start_time, end_time, ann_type)






training_filename = "../sample_data/vtach_beats.csv"
generate_training(training_filename)


# sig, fields = wfdb.rdsamp(data_path + sample_name)
# channel_sig = sig[:,channel_index]

# annotation = wfdb.rdann(ann_path + sample_name, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)[0]
# print annotation

# beats = get_beats(channel_sig, annotation)


# for beat in beats: 
# 	indices = beat[0]
# 	beat_sig = beat[1]
# 	time_vector = np.linspace(indices[0], indices[1], len(beat_sig))

# 	whole_sig = channel_sig[250*start:250*end]
# 	sig_time_vector = np.linspace(250*start, 250*end, len(whole_sig))

# 	annotation_y = [ channel_sig[ann_t] for ann_t in annotation ]

# 	plt.figure()
# 	plt.plot(sig_time_vector, whole_sig, 'b')
# 	plt.plot(time_vector, beat_sig, 'r')
# 	plt.plot(annotation, annotation_y, 'go')
# 	plt.show()



# print ""
# print annotation[0] / float(250.)