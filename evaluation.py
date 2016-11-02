
# coding: utf-8

# # Evaluation 

# In[78]:

from datetime                     import datetime
import numpy                      as np
import regular_activity           as regular
import load_annotations           as annotate
import invalid_sample_detection   as invalid
import parameters
import wfdb
import os
import json

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'config IPCompleter.greedy=True')


# ## Evaluation for sample names

# In[91]:

# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_sample(data_path, ann_path, ecg_ann_type, should_check_invalids=True,
                                         should_check_rr=True): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)            
            
            sig, fields = wfdb.rdsamp(data_path + sample_name)
            alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)
            
            is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type,
                                                   should_check_invalids, should_check_rr)
            
            # Classified as a true alarm if no regular activity 
            classified_true_alarm = not is_regular
            
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


# ## Evaluation with saving intermediate data

# In[93]:

def write_rr_file(data_path, ann_path, ecg_ann_type): 
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)            
            
            sig, fields = wfdb.rdsamp(data_path + sample_name)
            start, end, alarm_duration = invalid.get_start_and_end(fields)
            channels = fields['signame']
            num_channels = len(channels)
            ann_filename = "rr_" + ecg_ann_type + ".json"
            
            rr_dict = {}
            for channel_index in range(num_channels): 
                channel = fields['signame'][channel_index]
                # Ignore respiratory channel
                if channel == "RESP": 
                    continue

                rr_intervals, duration = annotate.get_channel_rr_intervals(ann_path, sample_name, channel_index, 
                                                                           fields, ecg_ann_type)
                rr_dict[channel] = rr_intervals.tolist()

def split_data(data1, data2): 
    only_in_data1 = set([])
    only_in_data2 = set([])
    
    for element in data1: 
        if element not in data2: 
            only_in_data1.add(element)
            
    for element in data2: 
        if element not in data1: 
            only_in_data2.add(element)
            
    return only_in_data1, only_in_data2

def read_rr_file(ann_path, ecg_ann_type):
    all_rr_dict = {}
    sample_name = ""
    ann_filename = "rr_" + ecg_ann_type + ".json"

    with open(os.path.join(ann_path, ann_filename), 'r') as f: 
        for line in f: 
            try:
                rr_intervals = json.loads(line)
                all_rr_dict[sample_name] = rr_intervals
            except Exception as e: 
                sample_name = line.strip()
                
    return all_rr_dict

    
def write_invalids_file(data_path): 
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)            
            
            sig, fields = wfdb.rdsamp(data_path + sample_name)
            start, end, alarm_duration = invalid.get_start_and_end(fields)
            
            invalids = invalid.calculate_invalids(data_path + sample_name, start, end)
            invalids_jsonifiable = { key : invalids[key].tolist() for key in invalids.keys() }
            
            with open(os.path.join(data_path, "invalids.json"), "a") as f: 
                f.write("\n" + sample_name + "\n")
                json.dump(invalids_jsonifiable, f)
                
def read_invalids_file(data_path):
    all_invalids_dict = {}
    sample_name = ""
    
    with open(os.path.join(data_path, "invalids.json"), 'r') as f: 
        for line in f: 
            try:
                invalids_dict = json.loads(line)
                all_invalids_dict[sample_name] = invalids_dict
            except Exception as e: 
                sample_name = line.strip()
                
    return all_invalids_dict


# In[99]:

# Generate confusion matrix for all samples given intermediate data
def generate_confusion_matrix_intermediate(data_path, ann_path, ecg_ann_type, 
                                           should_check_invalids=True, should_check_rr=True): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    all_invalids_dict = read_invalids_file(data_path)
    all_rr_dict = read_rr_file(ann_path, ecg_ann_type)
    
    for sample_name in all_invalids_dict.keys(): 
        invalids_dict = all_invalids_dict[sample_name]
        
        rr_intervals_dict = {}
        if sample_name in all_rr_dict.keys(): 
            rr_intervals_dict = all_rr_dict[sample_name]
        
        sig, fields = wfdb.rdsamp(data_path + sample_name)
        alarm_type, is_true_alarm = regular.check_gold_standard_classification(fields)
        start, end, alarm_duration = invalid.get_start_and_end(fields)        

        is_regular = regular.is_rr_invalids_regular(rr_intervals_dict, invalids_dict, alarm_duration,
                                                    should_check_invalids, should_check_rr)
            
        # Classified as a true alarm if no regular activity 
        classified_true_alarm = not is_regular

        if is_true_alarm and classified_true_alarm: 
            confusion_matrix["TP"].append(sample_name)
            
            if alarm_type == "Ventricular_Tachycardia": 
                print sample_name, alarm_type

        elif is_true_alarm and not classified_true_alarm: 
            confusion_matrix["FN"].append(sample_name)
            print "FALSE NEGATIVE: ", sample_name

        elif not is_true_alarm and classified_true_alarm: 
            confusion_matrix["FP"].append(sample_name)

        else: 
            confusion_matrix["TN"].append(sample_name)
            
    counts = { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }
                
    return counts, confusion_matrix


# ### Generate invalids, jqrs, and gqrs files

# In[21]:

data_path = "sample_data/challenge_training_data/"
ann_path = "sample_data/challenge_training_multiann/"

with open(os.path.join(data_path, "invalids.json"), 'w'): 
    pass
with open(os.path.join(ann_path, "rr_jqrs.json"), 'w'): 
    pass
with open(os.path.join(ann_path, "rr_gqrs.json"), 'w'): 
    pass


print "Writing jqrs file..."
write_rr_file(data_path, ann_path, 'jqrs')
print "Writing gqrs file..."
write_rr_file(data_path, ann_path, 'gqrs')
print "Writing invalids file..."
write_invalids_file(data_path)


# ## Evaluation stats

# In[53]:

def calc_sensitivity(counts): 
    tp = counts["TP"]
    fn = counts["FN"]
    return tp / float(tp + fn)
    
def calc_specificity(counts): 
    tn = counts["TN"]
    fp = counts["FP"]
    
    return tn / float(tn + fp)

def calc_ppv(counts): 
    tp = counts["TP"]
    fp = counts["FP"]
    return tp / float(tp + fp)

def calc_f1(counts): 
    sensitivity = calc_sensitivity(counts)
    ppv = calc_ppv(counts)
    
    return 2 * sensitivity * ppv / float(sensitivity + ppv)    


# In[55]:

Ventricular Tachycardiadef print_stats(counts): 
    sensitivity = calc_sensitivity(counts)
    specificity = calc_specificity(counts)
    ppv = calc_ppv(counts)
    f1 = calc_f1(counts)
    score = float(counts["TP"] + counts["TN"])/(counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"] * 5)

    print "counts: ", counts
    print "sensitivity: ", sensitivity
    print "specificity: ", specificity
    print "ppv: ", ppv
    print "f1: ", f1
    print "score: ", score


# In[100]:

if __name__ == '__main__': 
    data_path = 'sample_data/challenge_training_data/'
    ann_path = 'sample_data/challenge_training_multiann/'

#     generate_confusion_matrix_sample(data_path, ann_path, 'jqrs')
    
    counts_jqrs, confusion_matrix_jqrs = generate_confusion_matrix_intermediate(data_path, ann_path, 'jqrs')
#     counts_rr, confusion_matrix_rr = generate_confusion_matrix_intermediate(data_path, ann_path, 'jqrs', 
#                                                                                 False, # should_check_invalids
#                                                                                 True # should_check_rr
#                                                                                 )

#     print_stats(counts_jqrs)
#     print_stats(counts_rr)   


# In[ ]:



