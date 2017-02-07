
# coding: utf-8

# # Debugging

# In[2]:

from datetime                      import datetime
import invalid_sample_detection    as invalid
import load_annotations            as annotate
import regular_activity            as regular
import specific_arrhythmias        as arrhythmia
import numpy                       as np
import pipeline
import parameters
import os
import csv
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'


# ## Regular activity

# In[ ]:

def check_regular_activity_dir(data_path, ann_path, ecg_ann_type): 
    with open("sample_data/regular_activity.csv", "w") as f: 
        pass
        
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)
            is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type)
            
            with open("sample_data/regular_activity.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([sample_name, is_regular])

start = datetime.now()
check_regular_activity_dir(data_path, ann_path, ecg_ann_type)
print "time: ", datetime.now() - start


# In[ ]:

def compare_regular_activity(regular_activity_filename, fplesinger_filename): 
    confusion_matrix = {
        "FN": [], 
        "FP": []
    }
    
    with open(regular_activity_filename, "r") as f: 
        reader = csv.reader(f)
        
        for row in reader: 
            sample_name, is_regular = row[0], row[1] == "True"
            
            with open(fplesinger_filename, "r") as csvfile: 
                csvreader = csv.reader(csvfile)
                
                for csvrow in csvreader: 
                    if csvrow[0] == sample_name: 
                        regular_activity_array = [ x == '1' for x in csvrow[1:] ]
                        fplesinger_is_regular = any(regular_activity_array)
                        
                        if fplesinger_is_regular and not is_regular:
                            confusion_matrix["FN"].append(sample_name)
                            break
                            
                        elif not fplesinger_is_regular and is_regular: 
                            confusion_matrix["FP"].append(sample_name)
                            break
                            
    return confusion_matrix

regular_activity_filename = "sample_data/regular_activity.csv"
fplesinger_filename = "sample_data/fplesinger_data/output/regular-activity.csv"
print compare_regular_activity(regular_activity_filename, fplesinger_filename)


# In[ ]:



