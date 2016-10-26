
# coding: utf-8

# # Specific arrhythmia tests

# In[4]:

import invalid_sample_detection    as invalid
import load_annotations            as annotate
import parameters
import wfdb


# ## Asystole

# In[6]:

def calc_channel_asystole_score(ann_path, sample_name, ann_type, ann_fs, channel_start, channel_end): 
    current_start = channel_start
    current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    
    while current_end < channel_end: 
        
        annotation = wfdb.rdann(ann_path + sample_name, ann_type, sampfrom=current_start*ann_fs, sampto=current_end*ann_fs)
        
        if len(annotation[0]) > 0: 
            current_score = -1
        else: 
            current_score = 1
        
        current_start += parameters.ASYSTOLE_ROLLING_INCREMENT
        current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    
    


# In[5]:

def test_asystole(data_path, ann_path, sample_name): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    
    # Start and end given in seconds
    start, end, alarm_duration = invalid.get_start_and_end(fields)
    
    
data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
test_asystole(data_path, ann_path, "a555l")


# In[ ]:



