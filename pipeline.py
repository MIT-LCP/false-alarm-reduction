
# coding: utf-8

# # Overall pipeline

# In[2]:

from datetime                      import datetime
import invalid_sample_detection    as invalid
import load_annotations            as annotate
import regular_activity            as regular
import specific_arrhythmias        as arrhythmia
import numpy                       as np
import parameters
import os
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'


# In[3]:

# Returns true if alarm is classified as a true alarm
def is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, should_check_nan=False)    
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


# In[ ]:

if __name__ == '__main__': 
    counts, confusion_matrix = generate_confusion_matrix_dir(data_path, ann_path, ecg_ann_type)
    
    start = datetime.now() 
    print "counts: ", counts
    print "total time: ", datetime.now() - start


# In[ ]:



