
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
ecg_ann_type = 'jqrs'


# ## Classifying arrhythmia alarms

# In[3]:

# Returns true if alarm is classified as a true alarm
def is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, alarm_type, should_check_nan=False)    
    if is_regular: 
        return is_true_alarm, False
    
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
    
    classified_true_alarm = arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type)
    return is_true_alarm, classified_true_alarm


# In[4]:

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
            
            is_true_alarm, classified_true_alarm = is_classified_true_alarm(data_path, ann_path,
                                                                            sample_name, ecg_ann_type)

            if is_true_alarm and classified_true_alarm: 
                confusion_matrix["TP"].append(sample_name)
                
            elif is_true_alarm and not classified_true_alarm: 
                confusion_matrix["FN"].append(sample_name)
                print "FALSE NEGATIVE: ", filename
                
            elif not is_true_alarm and classified_true_alarm: 
                confusion_matrix["FP"].append(sample_name)
                
            else: 
                confusion_matrix["TN"].append(sample_name)
                
    counts = { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }
                
    return counts, confusion_matrix


# In[5]:

def print_by_type(false_negatives): 
    counts_by_type = {}
    for false_negative in false_negatives: 
        first = false_negative[0] 
        if first not in counts_by_type.keys(): 
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print counts_by_type


# In[ ]:

if __name__ == '__main__': 
#     start = datetime.now() 
#     counts_jqrs, confusion_matrix_jqrs = generate_confusion_matrix_dir(data_path, ann_path, 'jqrs')
#     print "total time: ", datetime.now() - start
    
#     evaluate.print_stats(counts_jqrs)
#     print_by_type(confusion_matrix_jqrs['FN'])

    start = datetime.now() 
    counts_gqrs, confusion_matrix_gqrs = generate_confusion_matrix_dir(data_path, ann_path, 'gqrs')
    print "total time: ", datetime.now() - start

    evaluate.print_stats(counts_gqrs)
    print_by_type(confusion_matrix_gqrs['FN'])


# ## Comparing classification with other algorithms

# In[9]:

def compare_algorithms(filename):
    with open(filename, "r") as f: 
        reader = csv.DictReader(f)
        authors = reader.fieldnames[1:]
        for row in reader: 
            sample_name = row["record name"]
            is_true_alarm, classified_true_alarm = is_classified_true_alarm(data_path, ann_path,
                                                                            sample_name, ecg_ann_type)
            

compare_algorithms("sample_data/answers.csv")


# In[ ]:

def generate_others_confusion_matrix(filename): 
    others_confusion_matrix = {}
    
    with open(filename, "r") as f: 
        reader = csv.DictReader(f)
        authors = reader.fieldnames[1:]
        for author in authors: 
            others_confusion_matrix[author] = { "TP": [], "FP": [], "FN": [], "TN": [] }
            
        for line in reader: 
            sample_name = line['record name']

