
# coding: utf-8

# # Overall pipeline

# In[27]:

from datetime                      import datetime
import invalid_sample_detection    as invalid
import evaluation                  as evaluate
import load_annotations            as annotate
import regular_activity            as regular
import specific_arrhythmias        as arrhythmia
import numpy                       as np
import parameters
import os
import csv
import json
import wfdb

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
fp_ann_path = 'sample_data/fplesinger_data/output/'
ecg_ann_type = 'gqrs'


# ## Classifying arrhythmia alarms

# In[33]:

# Returns true if alarm is classified as a true alarm
def classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)

    is_regular = regular.is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type, should_check_nan=False)    
    if is_regular:
        if verbose: 
            print sample_name + "with regular activity"
        return False
    
    alarm_type = sample_name[0]
    if alarm_type == "a": 
        arrhythmia_test = arrhythmia.test_asystole
    elif alarm_type == "b": 
        arrhythmia_test = arrhythmia.test_bradycardia
    elif alarm_type == "t": 
        arrhythmia_test = arrhythmia.test_tachycardia
    elif alarm_type == "v": 
        ann_path = fp_ann_path
        ecg_ann_type = 'fp'
        arrhythmia_test = arrhythmia.test_ventricular_tachycardia
    elif alarm_type == "f": 
        arrhythmia_test = arrhythmia.test_ventricular_flutter_fibrillation
    else: 
        raise Exception("Unknown arrhythmia alarm type")
    
    try: 
        classified_true_alarm = arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type, verbose)
        return classified_true_alarm

    except Exception as e: 
        print "sample_name: ", sample_name, e


# In[34]:

# Returns true if alarm is a true alarm
# Only for samples with known classification
def is_true_alarm(data_path, sample_name): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    alarm_type, true_alarm = regular.check_gold_standard_classification(fields)
    return true_alarm


# In[37]:

# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_dir(data_path, ann_path, fp_ann_path, ecg_ann_type): 
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }
    
    for filename in os.listdir(data_path):
        if filename.endswith(parameters.HEADER_EXTENSION):
            sample_name = filename.rstrip(parameters.HEADER_EXTENSION)
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type)

            matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
            confusion_matrix[matrix_classification].append(sample_name)
            if matrix_classification == "FN": 
                print "FALSE NEGATIVE: ", filename
                
    return confusion_matrix


def get_confusion_matrix_classification(true_alarm, classified_true_alarm): 
    if true_alarm and classified_true_alarm: 
        matrix_classification = "TP"

    elif true_alarm and not classified_true_alarm: 
        matrix_classification = "FN"

    elif not true_alarm and classified_true_alarm: 
        matrix_classification = "FP"

    else: 
        matrix_classification = "TN"

    return matrix_classification


# ## Printing and calculating counts

# In[36]:

def print_by_type(false_negatives): 
    counts_by_type = {}
    for false_negative in false_negatives: 
        first = false_negative[0] 
        if first not in counts_by_type.keys(): 
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print counts_by_type
    
    
def print_by_arrhythmia(confusion_matrix, arrhythmia_prefix): 
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys(): 
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    print counts_by_arrhythmia
    
def get_counts(confusion_matrix): 
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }


# ## Run pipeline

# In[ ]:

def run(data_path, ann_path, filename, ecg_ann_type):
    if ecg_ann_type == "fp": 
        ann_path = fp_ann_path
    print "ecg_ann_type: ", ecg_ann_type, " ann_path: ", ann_path
    
    start = datetime.now() 
    confusion_matrix_gqrs = generate_confusion_matrix_dir(data_path, ann_path, fp_ann_path, ecg_ann_type)
#     print "confusion matrix: ", confusion_matrix_gqrs
    print "total time: ", datetime.now() - start
    
    with open(filename, "w") as f: 
        json.dump(confusion_matrix_gqrs, f)

def read_json(filename): 
    with open(filename, "r") as f: 
        dictionary = json.load(f)
        
    return dictionary

print datetime.now()
write_filename = "sample_data/pipeline_fpinvalids_vtachfpann.json"
ecg_ann_type = "gqrs"
run(data_path, ann_path, write_filename, ecg_ann_type)


# In[32]:

if __name__ == '__main__': 
    print "GQRS"
    gqrs_matrix = read_json("sample_data/pipeline_gqrs.json")
    counts_gqrs = get_counts(gqrs_matrix)
    evaluate.print_stats(counts_gqrs)
    print_by_type(gqrs_matrix['FN'])
    
    fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
    print "missed true positives: ", get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TP")
    print "missed true negatives: ", get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TN")
    
    
    print "\nFP"
    fp_matrix = read_json("sample_data/pipeline_fp.json")
    counts_fp = get_counts(fp_matrix)
    evaluate.print_stats(counts_fp)
    print_by_type(fp_matrix['FN'])

    
    print "\nFP invalids with GQRS"
    fpinvalids_matrix = read_json("sample_data/pipeline_fpinvalids.json")
    counts_fpinvalids = get_counts(fpinvalids_matrix)
    evaluate.print_stats(counts_fpinvalids)
    print_by_type(fpinvalids_matrix['FN'])
    
    missed_true_negatives = get_missed(fpinvalids_matrix, fplesinger_confusion_matrix, "TN")
    print "missed true positives: ", get_missed(fpinvalids_matrix, fplesinger_confusion_matrix, "TP")
    print "missed true negatives: ", missed_true_negatives
    print_by_type(missed_true_negatives)
    print len(missed_true_negatives)
    
    
    print "\nFP invalids with GQRS without abp test in vtach"
    fpinvalids_without_vtach_abp = read_json("sample_data/pipeline_fpinvalids_novtachabp.json")
    counts_fpinvalids_without_vtach_abp = get_counts(fpinvalids_without_vtach_abp)
    evaluate.print_stats(counts_fpinvalids_without_vtach_abp)
    print_by_type(fpinvalids_without_vtach_abp['FN'])
    
#     print_by_type(gqrs_matrix['FN'])
#     print_by_arrhythmia(confusion_matrix_gqrs, 'v')
    
#     fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
#     print "missed true positives: ", get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TP")
#     print "missed true negatives: ", get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TN")


# ## Comparing classification with other algorithms

# In[21]:

def generate_others_confusion_matrices(filename, data_path): 
    others_confusion_matrices = {}
    
    with open(filename, "r") as f: 
        reader = csv.DictReader(f)
        authors = reader.fieldnames[1:]
        for author in authors: 
            others_confusion_matrices[author] = { "TP": [], "FP": [], "FN": [], "TN": [] }
            
        for line in reader: 
            sample_name = line['record name']
            true_alarm = is_true_alarm(data_path, sample_name)
            
            for author in authors: 
                classified_true_alarm = line[author] == '1'
                matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
                
                others_confusion_matrices[author][matrix_classification].append(sample_name)
    
    return others_confusion_matrices
                
    
filename = "sample_data/answers.csv"
others_confusion_matrices = generate_others_confusion_matrices(filename, data_path)    


# In[7]:

for author in others_confusion_matrices.keys(): 
    other_confusion_matrix = others_confusion_matrices[author]
    print author
    counts = get_counts(other_confusion_matrix)
    evaluate.print_stats(counts)
    print_by_type(other_confusion_matrix['FN'])


# In[23]:

def get_missed(confusion_matrix, other_confusion_matrix, classification): 
    missed = []
    
    for sample in other_confusion_matrix[classification]: 
        if sample not in confusion_matrix[classification]: 
            missed.append(sample)
            
    return missed
    
# fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
# print "missed true positives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TP")
# print "missed true negatives: ", get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TN")


# In[ ]:



