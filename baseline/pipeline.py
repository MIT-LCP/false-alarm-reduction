
# coding: utf-8

# # Overall pipeline

from datetime                      import datetime
from baseline_algorithm            import * 
import numpy                       as np
import parameters
import os
import csv
import json
import wfdb


# Returns true if alarm is a true alarm
# Only for samples with known classification
def is_true_alarm(data_path, sample_name): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    true_alarm = fields['comments'][1] == 'True alarm'
    return true_alarm


# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_dir(data_path, ann_path, fp_ann_path, ecg_ann_type): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)

            # sig, fields = wfdb.srdsamp(data_path + sample_name)
            # if "II" not in fields['signame']: 
            #     continue
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type)

            matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
            confusion_matrix[matrix_classification].append(sample_name)
            if matrix_classification == "FN": 
                print "FALSE NEGATIVE: ", filename
                
    return confusion_matrix


def get_confusion_matrix_classification(true_alarm, classified_true_alarm): 
    if true_alarm and classified_true_alarm: 
        matrix_classification = "TP"

    elif true_alarm and not classified_true_alarm: 
        matrix_classification = "FN"

    elif not true_alarm and classified_true_alarm: 
        matrix_classification = "FP"

    else: 
        matrix_classification = "TN"

    return matrix_classification


# ## Printing and calculating counts

def print_by_type(false_negatives): 
    counts_by_type = {}
    for false_negative in false_negatives: 
        first = false_negative[0] 
        if first not in counts_by_type.keys(): 
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print counts_by_type
    
    
def print_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    print counts_by_arrhythmia
    
def get_counts(confusion_matrix): 
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }


def calc_sensitivity(counts): 
    tp = counts["TP"]
    fn = counts["FN"]
    return tp / float(tp + fn)
    
def calc_specificity(counts): 
    tn = counts["TN"]
    fp = counts["FP"]
    
    return tn / float(tn + fp)

def calc_ppv(counts): 
    tp = counts["TP"]
    fp = counts["FP"]
    return tp / float(tp + fp)

def calc_f1(counts): 
    sensitivity = calc_sensitivity(counts)
    ppv = calc_ppv(counts)
    
    return 2 * sensitivity * ppv / float(sensitivity + ppv)    

def print_stats(counts): 
    sensitivity = calc_sensitivity(counts)
    specificity = calc_specificity(counts)
    ppv = calc_ppv(counts)
    f1 = calc_f1(counts)
    score = float(counts["TP"] + counts["TN"])/(counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"] * 5)

    print "counts: ", counts
    print "sensitivity: ", sensitivity
    print "specificity: ", specificity
    print "ppv: ", ppv
    print "f1: ", f1
    print "score: ", score


# ## Run pipeline

def run(data_path, ann_path, fp_ann_path, filename, ecg_ann_type):
    if ecg_ann_type == "fp": 
        ann_path = fp_ann_path
    print "ecg_ann_type: ", ecg_ann_type, " ann_path: ", ann_path
    
    start = datetime.now() 
    matrix = generate_confusion_matrix_dir(data_path, ann_path, fp_ann_path, ecg_ann_type)
    print "confusion matrix: ", matrix
    print "total time: ", datetime.now() - start
    
    with open(filename, "w") as f: 
        json.dump(matrix, f)

def read_json(filename): 
    with open(filename, "r") as f: 
        dictionary = json.load(f)
        
    return dictionary


if __name__ == '__main__': 
    run(data_path, ann_path, "", write_filename, ecg_ann_type)

    matrix = read_json(write_filename)
    counts = get_counts(matrix)
    print_stats(counts)

