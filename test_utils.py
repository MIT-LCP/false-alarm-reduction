from datetime import datetime
import utils
import os
import parameters
import json
import wfdb

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

            print("Sample name: ", sample_name)
            
            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = utils.classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type)

            matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
            confusion_matrix[matrix_classification].append(sample_name)
            if matrix_classification == "FN": 
                print("FALSE NEGATIVE: ", filename)
                
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


# Returns true if alarm is a true alarm
# Only for samples with known classification
def is_true_alarm(data_path, sample_name): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    return "True" in fields['comments'][1]
    

def run(data_path, ann_path, fp_ann_path, filename, ecg_ann_type):
    if ecg_ann_type == "fp": 
        ann_path = fp_ann_path
    
    start = datetime.now() 
    print("start: ", start)
    confusion_matrix = generate_confusion_matrix_dir(data_path, ann_path, fp_ann_path, ecg_ann_type)
    print("total time: ", datetime.now() - start)
    
    with open(filename, "w") as f: 
        json.dump(confusion_matrix, f)

def read_json(filename): 
    with open(filename, "r") as f: 
        dictionary = json.load(f)
        
    return dictionary


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


# In[8]:

def print_stats(counts): 
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

def get_counts(confusion_matrix): 
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }

if __name__ == '__main__':
	data_path = 'sample_data/challenge_training_data/'
	ann_path = 'sample_data/challenge_training_multiann/'
	fp_ann_path = 'sample_data/fplesinger_data/output/'
	ecg_ann_type = 'gqrs'

	write_filename = "sample_data/test_utils.json"
	# run(data_path, ann_path, fp_ann_path, write_filename, ecg_ann_type)

	matrix = read_json(write_filename)
	counts = get_counts(matrix)
	print_stats(counts)

	# classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type)