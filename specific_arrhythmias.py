
# coding: utf-8

# # Specific arrhythmia tests

# In[4]:

import invalid_sample_detection    as invalid
import load_annotations            as annotate
import parameters
import wfdb


# ## Asystole

# In[18]:

def calc_channel_asystole_score(ann_path, sample_name, sig, fields, ann_type, channel_start, channel_end,
                                channel): 
    current_start = channel_start
    current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    cumulative_score = 0
    
    while current_end < channel_end: 
        channel_type = invalid.get_channel_type(channel)
        annotation, ann_fs = annotate.get_annotation_annfs(ann_path + sample_name, ann_type, current_start,
                                                           current_end, channel_type)
        
        if len(annotation[0]) > 0: 
            current_score = -1
        else: 
            current_score = 1
            
        invalids = invalid.calculate_invalids_sig(sig, fields, current_start, current_end)
        cval = invalid.calculate_cval(invalids)        
        current_score *= cval[channel]
        
        cumulative_score += current_score
        
        current_start += parameters.ASYSTOLE_ROLLING_INCREMENT
        current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    
    return cumulative_score   


# In[ ]:

def test_asystole(data_path, ann_path, sample_name, ecg_ann_type): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    channels = fields['signame']
        
    # Start and end given in seconds
    start, end, alarm_duration = invalid.get_start_and_end(fields)
    
    overall_score = 0    
    for channel_index, channel in zip(range(len(channels)), channels): 
        ann_type = annotate.get_ann_type(channel, channel_index, ecg_ann_type)
        channel_score = calc_channel_asystole_score(ann_path, sample_name, sig, fields, ann_type, start, end, 
                                                    channel)
        
        overall_score += channel_score
        
    return overall_score > 0
    
    
data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
sample_name = "a203l"
ecg_ann_type = 'jqrs'
test_asystole(data_path, ann_path, sample_name, ecg_ann_type)


# In[ ]:



