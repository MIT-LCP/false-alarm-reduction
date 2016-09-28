
# coding: utf-8

# # Regular activity test
Questions: 
1) Annotations are only generated based on the first lead. What happens when the first lead is noisy/invalid and the other channels aren't but still don't get annotated properly? 
2) Annotations for when the alarm was triggered? --> do regular activity test on blocks of 10s or on the 10s when the alarm was triggered? 

b695l: all channels without regular activity except ppg which isn't being properly annotated (70000-72500)
# In[2]:

import invalid_sample_detection    as invalid
import load_annotations            as annotate
import matplotlib.pyplot           as plt
import numpy                       as np
import parameters
import wfdb
import math

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'config IPCompleter.greedy=True')


# ## RR intervals tests

# In[9]:

def check_rr_stdev(rr_intervals): 
    numpy_rr_intervals = np.array(rr_intervals)
    stdev = np.std(numpy_rr_intervals)
    if stdev > parameters.RR_STDEV: 
        return False
    return True

def check_heart_rate(rr_intervals, start, end): 
    start_time = float(start) / parameters.FS
    end_time = float(end) / parameters.FS
    time_length = end_time - start_time
    hr = len(rr_intervals) / time_length
    
    if hr > parameters.HR_MAX or hr < parameters.HR_MIN: 
        return False
    return True

def check_sum_rr_intervals(rr_intervals, start, end): 
    start_time = float(start) / parameters.FS
    end_time = float(end) / parameters.FS
    min_sum = (end_time - start_time) - parameters.RR_MIN_SUM_DIFF
    
    rr_sum = sum(rr_intervals)
    if rr_sum < min_sum: 
        return False
    return True    

def check_num_rr_intervals(rr_intervals): 
    if len(rr_intervals) < parameters.MIN_NUM_RR_INTERVALS: 
        return False
    return True    


# ## Invalids tests

# In[11]:

def check_invalids(invalids, start, end): 
    start_block = int(float(start) / parameters.FS * parameters.BLOCK_LENGTH)
    end_block = int(math.ceil(float(end) / parameters.FS * parameters.BLOCK_LENGTH))
    
    block_invalids_sum = 0
    for block_index in range(start_block, end_block + 1): 
        if block_index >= len(invalids): 
            raise Exception("Block_index " + str(block_index) + " and len(invalids) " + str(len(invalids)))
            
        block_invalids_sum += invalids[block_index]
    
    if block_invalids_sum > 0: 
        return False
    return True


# ## Putting it all together

# In[5]:

def check_interval_regular_activity(rr_intervals, invalids, start, end): 
    stdev_check = check_rr_stdev(rr_intervals)
    hr_check = check_heart_rate(rr_intervals, start, end)
    sum_check = check_sum_rr_intervals(rr_intervals, start, end)
    num_check = check_num_rr_intervals(rr_intervals)
    invalids_check = check_invalids(invalids, start, end)
    
    return stdev_check and hr_check and sum_check and num_check and invalids_check

def check_channel_regular_activity(data_path, ann_path, sample_name, ann_type, start, end): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    channels = fields['signame']
    num_channels = len(channels)
    invalids = invalid.calculate_invalids_standard(data_path + sample_name, start, end)
    print "invalids: ", invalids

    regular_activity = False
    for channel_index in range(num_channels): 
        if channels[channel_index] == "RESP": 
            continue
        print "channel: ", channels[channel_index]
        
        rr_intervals = annotate.calculate_rr_intervals_standard(ann_path + sample_name, channel_index, ann_type, start, end)
        print "rr_intervals: ", rr_intervals
        
        is_regular = check_interval_regular_activity(rr_intervals, invalids, start, end)
        print "is_regular: ", is_regular
        
        if is_regular: 
            return True
    return False


# In[12]:

if __name__ == '__main__': 
    data_path = 'sample_data/challenge_training_data/'
    ann_path = 'sample_data/challenge_training_ann/'
    sample_name = 'v131l'
    
    ann_type = 'jqrs'
    start = 73750 # in sample number
    end = 76250 # in sample number
    channel = 0    
    
    time_window = 10 # in seconds
    
    print check_channel_regular_activity(data_path, ann_path, sample_name, ann_type, start, end)


# In[ ]:



