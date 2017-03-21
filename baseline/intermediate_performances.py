import sys
from parameters                 import *
sys.path.append('../dtw')
from baseline_algorithm         import classify_alarm
from utils                      import *  # utils from dtw file
import wfdb
import os


def run_by_arrhythmia(data_path, ann_path, fp_ann_path, ecg_ann_type, arrhythmia, json_filename): 
    matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }

    for filename in os.listdir(data_path):
        if filename.endswith(HEADER_EXTENSION):
            sample_name = filename.rstrip(HEADER_EXTENSION)

            if sample_name[0] != arrhythmia: 
                continue
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type)

            classification = get_matrix_classification(true_alarm, classified_true_alarm)
            matrix[classification].append(sample_name)

    write_json(matrix, json_filename)


if __name__ == '__main__': 
    data_path = '../sample_data/challenge_training_data/'
    ann_path = '../sample_data/challenge_training_multiann/'
    fp_ann_path = '../sample_data/fplesinger_data/'

    arrhythmias = {
        'a': 'asystole', 
        'b': 'bradycardia',
        't': 'tachycardia',
        'f': 'vfib', 
        'v': 'vtach' 
    }
    # ann_types = ['gqrs', 'jqrs', 'fp']

    # for ecg_ann_type in ann_types: 
    #     for arrhythmia in arrhythmias.keys(): 
    #         json_filename = "../sample_data/" + arrhythmias[arrhythmia] + "_" + ecg_ann_type + ".json"


    ecg_ann_type = 'gqrs'

    for arrhythmia in arrhythmias.keys(): 
        json_filename = "../sample_data/" + arrhythmias[arrhythmia] + "_" + ecg_ann_type + ".json"

        run_by_arrhythmia(data_path, ann_path, fp_ann_path, ecg_ann_type, arrhythmia, json_filename)

        matrix = read_json(json_filename)
        counts = { key : len(matrix[key]) for key in matrix.keys()}

        print "\nARRHYTHMIA: ", arrhythmias[arrhythmia]
        print "accuracy:", get_classification_accuracy(matrix)
        print "score:", get_score(matrix)
        print_stats(counts)


