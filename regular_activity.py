
# coding: utf-8

# # Regular activity test

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

# In[14]:

def check_rr_stdev(rr_intervals): 
    numpy_rr_intervals = np.array(rr_intervals)
    stdev = np.std(numpy_rr_intervals)
    if stdev > parameters.RR_STDEV: 
        return False
    return True

def check_heart_rate(rr_intervals, alarm_duration): 
    hr = len(rr_intervals) / float(alarm_duration) * parameters.NUM_SECS_IN_MIN
        
    if hr > parameters.HR_MAX or hr < parameters.HR_MIN: 
        return False
    return True

def check_sum_rr_intervals(rr_intervals, alarm_duration): 
    min_sum = alarm_duration - parameters.RR_MIN_SUM_DIFF
        
    rr_sum = sum(rr_intervals)
    if rr_sum < min_sum: 
        return False
    return True    

def check_num_rr_intervals(rr_intervals): 
    if len(rr_intervals) < parameters.MIN_NUM_RR_INTERVALS: 
        return False
    return True    


# ## Invalids tests

# In[4]:

def check_invalids(invalids, channel): 
    if channel not in invalids.keys(): 
        raise Exception("Unknown channel")
    
    block_invalids_sum = sum(invalids[channel])
    if block_invalids_sum > 0: 
        return False
    return True


# ## Putting it all together

# In[5]:

def check_interval_regular_activity(rr_intervals, invalids, alarm_duration, channel): 
    stdev_check = check_rr_stdev(rr_intervals)
    hr_check = check_heart_rate(rr_intervals, alarm_duration)
    sum_check = check_sum_rr_intervals(rr_intervals, alarm_duration)
    num_check = check_num_rr_intervals(rr_intervals)
    invalids_check = check_invalids(invalids, channel)
    
    all_checks = [stdev_check, hr_check, sum_check, num_check, invalids_check]
    if not all(all_checks): 
        print "stdev_check: ", stdev_check
        print "hr_check: ", hr_check
        print "sum_check: ", sum_check
        print "num_check: ", num_check
        print "invalids_check: ", invalids_check
    
    return all(all_checks)

def check_channel_regular_activity(data_path, ann_path, sample_name, ann_type, start, end, alarm_duration): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    
    channels = fields['signame']
    num_channels = len(channels)
    fs = fields['fs']
        
    invalids = invalid.calculate_invalids_standard(data_path + sample_name, start, end)
    print "invalids: ", invalids

    # Eventually calculate RR intervals for all channels and move block into for loop
    channel0_rr_intervals = annotate.calculate_rr_intervals(ann_path + sample_name, 0, fs, ann_type, start, end)
    print "rr_intervals: ", channel0_rr_intervals
    
    for channel_index in range(num_channels): 
        channel = channels[channel_index]
        if channel == "RESP": 
            continue
        print "\nchannel: ", channel
        is_regular = check_interval_regular_activity(channel0_rr_intervals, invalids, alarm_duration, channel)
        
        if is_regular: 
            return True
    return False


# In[6]:

def get_start_and_end(data_path, sample_name): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)

    fs = fields['fs']
    alarm_type = fields['comments'][0]
    if alarm_type not in parameters.TESTED_BLOCK_LENGTHS: 
        raise Exception("Unrecognized alarm type")
    tested_block_length = parameters.TESTED_BLOCK_LENGTHS[alarm_type]
    
    end = fs * parameters.ALARM_TIME # in sample number, alarm always sounded at 300th second
    start = end - fs * tested_block_length # in sample number
    
    return (start, end, tested_block_length)

def is_classified_correctly(is_true_alarm, alarm_classification, is_regular): 
    is_classified_true_alarm = not is_regular
    matches = is_true_alarm is is_classified_true_alarm
    if matches: 
        return "\nMatches! Alarm was a " + alarm_classification.lower()
    else:
        return "\n" + alarm_classification + " classified as a " + str(is_classified_true_alarm).lower() + " alarm"


# In[15]:

if __name__ == '__main__': 
    data_path = 'sample_data/challenge_training_data/'
    ann_path = 'sample_data/challenge_training_ann/'
    sample_name = 'v111l'
    ann_type = 'jqrs'
    
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, alarm_classification = fields['comments']
    if "True" in alarm_classification: 
        is_true_alarm = True
    else:
        is_true_alarm = False
            
    start, end, alarm_duration = get_start_and_end(data_path, sample_name)    
    is_regular = check_channel_regular_activity(data_path, ann_path, sample_name, ann_type, start, end, alarm_duration)
    
    print is_classified_correctly(is_true_alarm, alarm_classification, is_regular)
    
    annotate.plot_annotations(data_path, ann_path, sample_name, ['jqrs', 'gqrs'], 0, fields['fs'], start, end)


# In[ ]:



