
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
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'


# In[4]:

# Returns true if alarm is classified as a true alarm
def is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, should_check_nan=False)    
    if is_regular: 
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
        
    return arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type)

sample_name = "f697l"
print is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type)


# In[ ]:



