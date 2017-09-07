import sys
from utils import *

gqrs_filename = "../sample_data/vtach_gqrs.json"
jqrs_filename = "../sample_data/vtach_jqrs.json"
dtw_filename = "../sample_data/dtw_fromfile.json"

gqrs_matrix = read_json(gqrs_filename)
jqrs_matrix = read_json(jqrs_filename)
dtw_matrix = read_json(dtw_filename)

gqrs_vtach_counts, gqrs_vtach_matrix = get_by_arrhythmia(gqrs_matrix, 'v')
jqrs_vtach_counts, jqrs_vtach_matrix = get_by_arrhythmia(jqrs_matrix, 'v')

samples = {}
for classification in dtw_matrix: 
	for sample_name in dtw_matrix[classification]: 
		is_true_alarm = classification == "TP" or classification == "FN"
		samples[sample_name] = is_true_alarm


def is_classified_true_alarm(matrix, sample_name): 
	for classification in matrix: 
		if sample_name in matrix[classification]: 
			return classification == "FP" or classification == "TP"
	return None



# Strategy 1: Baseline, defer to DTW if baseline thinks the alarm is a false alarm
matrix = {
	'FP': [],
	'TP': [],
	'TN': [],
	'FN': []
}

for sample_name in samples.keys(): 
	is_classified_true_alarm_gqrs = is_classified_true_alarm(gqrs_vtach_matrix, sample_name)
	is_classified_true_alarm_jqrs = is_classified_true_alarm(jqrs_vtach_matrix, sample_name)
	is_classified_true_alarm_dtw = is_classified_true_alarm(dtw_matrix, sample_name)

	# Mark as negative
	if not is_classified_true_alarm_jqrs: 
		is_classified_true_alarm_combined = False

	# elif not is_classified_true_alarm_dtw:
	# 	is_classified_true_alarm_combined = False

	else:  # Defer to DTW 
		is_classified_true_alarm_combined = is_classified_true_alarm_gqrs

	classification = get_matrix_classification(samples[sample_name], is_classified_true_alarm_combined)

	matrix[classification].append(sample_name)


counts = { key : len(matrix[key]) for key in matrix.keys() }

print "accuracy:", get_classification_accuracy(matrix)
print "score:", get_score(matrix)
print_stats(counts)

print ""

# Strategy 2: DTW, defer to baseline algorithm if DTW thinks the alarm is a false alarm
matrix = {
	'FP': [],
	'TP': [],
	'TN': [],
	'FN': []
}

for sample_name in samples.keys(): 
	is_classified_true_alarm_gqrs = is_classified_true_alarm(gqrs_vtach_matrix, sample_name)
	is_classified_true_alarm_dtw = is_classified_true_alarm(dtw_matrix, sample_name)

	# Mark as positive
	if is_classified_true_alarm_dtw: 
		is_classified_true_alarm_combined = True

	else: # Defer to gqrs
		is_classified_true_alarm_combined = is_classified_true_alarm_gqrs

	classification = get_matrix_classification(samples[sample_name], is_classified_true_alarm_combined)

	matrix[classification].append(sample_name)


counts = { key : len(matrix[key]) for key in matrix.keys() }

print "accuracy:", get_classification_accuracy(matrix)
print "score:", get_score(matrix)
print_stats(counts)





