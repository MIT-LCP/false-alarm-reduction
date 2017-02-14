
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
fp_ann_path = 'sample_data/fplesinger_data/output/'
ecg_ann_type = 'gqrs'


# ## Regular activity

# In[18]:

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

ecg_ann_type = "gqrs"
start = datetime.now()
check_regular_activity_dir(data_path, ann_path, ecg_ann_type)
print "time: ", datetime.now() - start


# In[31]:

def compare_regular_activity(regular_activity_filename, fplesinger_filename): 
    confusion_matrix = {
        "FN": [], 
        "FP": []
    }
        
    with open(regular_activity_filename, "r") as f: 
        reader = csv.reader(f)
        
        # For each sample in our algorithm regular activity file
        for row in reader: 
            sample_name, is_regular = row[0], row[1] == "True"
            
            # Find the associated sample in the fplesinger regular activity file
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


regular_activity_filename = "sample_data/regular_activity_gqrs.csv"
fplesinger_filename = "sample_data/fplesinger_data/output/regular-activity.csv"
print compare_regular_activity(regular_activity_filename, fplesinger_filename)


# None of the 162 samples classified with regular activity by our algorithm were true alarms (so the regular activity test did not cause any false negatives). Interestingly, the fplesinger algorithm also classified 162 samples with regular activity, although a total of 64 samples were classified differently between the two algorithms (32 classified as regular activity by our algorithm and not the fplesinger algorithm and 32 classified as regular activity by the fplesinger algorithm and not our algorithm). 

# ## Annotations

# In[ ]:

def check_annotations_dir(data_path, ann_path, fp_ann_path, ecg_ann_type): 
    with open("sample_data/annotations_fp_gqrs.csv", "w") as f: 
        pass
        
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)
            sig, fields = wfdb.rdsamp(data_path + sample_name)
            channels = fields['signame']
            
            for channel_index, channel_name in zip(range(channels), channels): 
                fp_ann_type = annotate.get_ann_type(channel_name, channel_index, "fp")
                ann_type = annotate.get_ann_type(channel_name, channel_index, ecg_ann_type)
                try: 
                    fp_ann = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
                except Exception as e: 
                    annotation = []
                    print e
            
            
            with open("sample_data/regular_activity.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([sample_name, is_regular])

ecg_ann_type = "gqrs"
start = datetime.now()
check_regular_activity_dir(data_path, ann_path, ecg_ann_type)
print "time: ", datetime.now() - start

