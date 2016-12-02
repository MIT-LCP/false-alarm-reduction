
# coding: utf-8

# # Overall pipeline

# In[2]:

from datetime                      import datetime
import invalid_sample_detection    as invalid
import evaluation                  as evaluate
import load_annotations            as annotate
import regular_activity            as regular
import specific_arrhythmias        as arrhythmia
import numpy                       as np
import parameters
import os
import csv
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'


# ## Classifying arrhythmia alarms

# In[4]:

# Returns true if alarm is classified as a true alarm
def is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, alarm_type, should_check_nan=False)    
    if is_regular:
        if verbose: 
            print sample_name + "with regular activity"
        return False
    
    if alarm_type == "Asystole": 
        arrhythmia_test = arrhythmia.test_asystole
    elif alarm_type == "Bradycardia": 
        arrhythmia_test = arrhythmia.test_bradycardia
    elif alarm_type == "Tachycardia": 
        arrhythmia_test = arrhythmia.test_tachycardia
    elif alarm_type == "Ventricular_Tachycardia": 
        arrhythmia_test = arrhythmia.test_ventricular_tachycardia
    elif alarm_type == "Ventricular_Flutter_Fib": 
        arrhythmia_test = arrhythmia.test_ventricular_flutter_fibrillation
    else: 
        raise Exception("Unknown arrhythmia alarm type")
    
    classified_true_alarm = arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type, verbose)
    return classified_true_alarm


def is_true_alarm(data_path, sample_name): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, true_alarm = regular.check_gold_standard_classification(fields)
    return true_alarm


# In[28]:

# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_dir(data_path, ann_path, ecg_ann_type): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type)

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


# In[5]:

def print_by_type(false_negatives): 
    counts_by_type = {}
    for false_negative in false_negatives: 
        first = false_negative[0] 
        if first not in counts_by_type.keys(): 
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print counts_by_type
    
def get_counts(confusion_matrix): 
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }


# In[16]:

# if __name__ == '__main__': 
#     start = datetime.now() 
#     confusion_matrix_gqrs = generate_confusion_matrix_dir(data_path, ann_path, 'gqrs')
#     counts_gqrs = get_counts(confusion_matrix_gqrs)
#     print "total time: ", datetime.now() - start

#     evaluate.print_stats(counts_gqrs)
#     print_by_type(confusion_matrix_gqrs['FN'])


# ## Comparing classification with other algorithms

# In[23]:

def generate_others_confusion_matrices(filename, data_path): 
    others_confusion_matrices = {}
    
    with open(filename, "r") as f: 
        reader = csv.DictReader(f)
        authors = reader.fieldnames[1:]
        for author in authors: 
            others_confusion_matrices[author] = { "TP": [], "FP": [], "FN": [], "TN": [] }
            
        for line in reader: 
            sample_name = line['record name']
            true_alarm = is_true_alarm(data_path, sample_name)
            
            for author in authors: 
                classified_true_alarm = line[author] == '1'
                matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
                
                others_confusion_matrices[author][matrix_classification].append(sample_name)
    
    return others_confusion_matrices
                
    
filename = "sample_data/answers.csv"
others_confusion_matrices = generate_others_confusion_matrices(filename, data_path)    


# In[24]:

for author in others_confusion_matrices.keys(): 
    other_confusion_matrix = others_confusion_matrices[author]
    print author
    counts = get_counts(other_confusion_matrix)
    evaluate.print_stats(counts)
    print_by_type(other_confusion_matrix['FN'])


# In[ ]:



