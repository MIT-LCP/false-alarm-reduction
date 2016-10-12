
# coding: utf-8

# # Evaluation 

# In[2]:

from datetime             import datetime
import numpy              as np
import regular_activity   as regular
import parameters
import wfdb
import os

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'config IPCompleter.greedy=True')


# ## Evaluation metrics

# In[3]:

def generate_confusion_matrix(data_path, ann_path, ecg_ann_type, should_check_invalids=True, should_check_rr=True): 
    confusion_matrix = {
        "TP": 0.,
        "FP": 0.,
        "FN": 0.,
        "TN": 0.
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
                confusion_matrix["TP"] += 1
                
            elif is_true_alarm and not classified_true_alarm: 
                confusion_matrix["FN"] += 1
                print "FALSE NEGATIVE: ", filename
                
            elif not is_true_alarm and classified_true_alarm: 
                confusion_matrix["FP"] += 1
                
            else: 
                confusion_matrix["TN"] += 1
                
    return confusion_matrix


# In[4]:

def calc_sensitivity(confusion_matrix): 
    tp = confusion_matrix["TP"]
    fn = confusion_matrix["FN"]
    return tp / (tp + fn)
    
def calc_specificity(confusion_matrix): 
    tn = confusion_matrix["TN"]
    fp = confusion_matrix["FP"]
    return tn / (tn + fp)

def calc_ppv(confusion_matrix): 
    tp = confusion_matrix["TP"]
    fp = confusion_matrix["FP"]
    return tp / (tp + fp)

def calc_f1(confusion_matrix): 
    sensitivity = calc_sensitivity(confusion_matrix)
    ppv = calc_ppv(confusion_matrix)
    
    return 2 * sensitivity * ppv / (sensitivity + ppv)    


# In[5]:

def print_stats(confusion_matrix): 
    sensitivity = calc_sensitivity(confusion_matrix)
    specificity = calc_specificity(confusion_matrix)
    ppv = calc_ppv(confusion_matrix)
    f1 = calc_f1(confusion_matrix)
    score = (confusion_matrix["TP"] + confusion_matrix["TN"])/(confusion_matrix["TP"] + confusion_matrix["FP"] +
                                                           confusion_matrix["TN"] + confusion_matrix["FN"] * 5)

    print "confusion_matrix: ", confusion_matrix
    print "sensitivity: ", sensitivity
    print "specificity: ", specificity
    print "ppv: ", ppv
    print "f1: ", f1
    print "score: ", score


# In[6]:

if __name__ == '__main__': 
    data_path = 'sample_data/challenge_training_data/'
    ann_path = 'sample_data/challenge_training_multiann/'

    start = datetime.now()
    confusion_matrix_jqrs = generate_confusion_matrix(data_path, ann_path, 'jqrs')
    print "time: ", datetime.now() - start
    
    start = datetime.now()
    confusion_matrix_gqrs = generate_confusion_matrix(data_path, ann_path, 'gqrs')
    print "time: ", datetime.now() - start
    
    start = datetime.now()
    confusion_matrix_rr = generate_confusion_matrix(data_path, ann_path, 'jqrs',
                                                    False, # should_check_invalids 
                                                    True # should_check_rr
                                                   )
    print "time: ", datetime.now() - start

    
    print "jqrs (both)"
    print_stats(confusion_matrix_jqrs)
    
    print "gqrs (both)"
    print_stats(confusion_matrix_gqrs)
    
    print "jqrs (only rr)"
    print_stats(confusion_matrix_rr)
    


# Results: 
# 
# invalids and rr
# confusion_matrix: {'FP': 373.0, 'TN': 79.0, 'FN': 1.0, 'TP': 293.0}
# sensitivity:  0.996598639456
# specificity:  0.174778761062
# ppv:  0.43993993994
# f1:  0.610416666667
# score:  0.496
# 
# invalids
# confusion_matrix:  {'FP': 7.0, 'TN': 445.0, 'FN': 283.0, 'TP': 11.0}
# sensitivity:  0.0374149659864
# specificity:  0.984513274336
# ppv:  0.611111111111
# f1:  0.0705128205128
# score:  0.242811501597
# 
# rr
# confusion_matrix:  {'FP': 373.0, 'TN': 79.0, 'FN': 1.0, 'TP': 293.0}
# sensitivity:  0.996598639456
# specificity:  0.174778761062
# ppv:  0.43993993994
# f1:  0.610416666667
# score:  0.496

# In[ ]:

with open("rr_intervals.txt", "w") as f: 
    

