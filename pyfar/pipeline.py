from __future__ import print_function

from datetime                       import datetime
import numpy                        as np
from baseline_algorithm             import *
from parameters                     import *
import os
import csv
import json
import wfdb

# ## Classifying arrhythmia alarms

# Returns true if alarm is a true alarm
# Only for samples with known classification
def is_true_alarm(data_path, sample_name):
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    true_alarm = fields['comments'][1] == 'True alarm'
    return true_alarm

# Generate confusion matrix for all samples given sample name/directory
def generate_confusion_matrix_dir(data_path, ann_path, ecg_ann_type):
    confusion_matrix = {
        "TP": [],
        "FP": [],
        "FN": [],
        "TN": []
    }

    for filename in os.listdir(data_path):
        if filename.endswith(HEADER_EXTENSION):
            sample_name = filename.rstrip(HEADER_EXTENSION)

            if sample_name[0] != 'v':
                continue

            print("sample name:  {}".format(sample_name))

            # sig, fields = wfdb.srdsamp(data_path + sample_name)
            # if "II" not in fields['signame']:
            #     continue

            true_alarm = is_true_alarm(data_path, sample_name)
            classified_true_alarm = classify_alarm(data_path, ann_path, sample_name, ecg_ann_type)

            matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)
            confusion_matrix[matrix_classification].append(sample_name)
            if matrix_classification == "FN":
                print("FALSE NEGATIVE:  {}".format(filename))

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

def print_by_type(false_negatives):
    counts_by_type = {}
    for false_negative in false_negatives:
        first = false_negative[0]
        if first not in counts_by_type.keys():
            counts_by_type[first] = 0
        counts_by_type[first] += 1

    print(counts_by_type)


def print_by_arrhythmia(confusion_matrix, arrhythmia_prefix):
    counts_by_arrhythmia = {}
    for classification_type in confusion_matrix.keys():
        sample_list = [ sample for sample in confusion_matrix[classification_type] if sample[0] == arrhythmia_prefix]
        counts_by_arrhythmia[classification_type] = (len(sample_list), sample_list)

    print(counts_by_arrhythmia)

def get_counts(confusion_matrix):
    return { key : len(confusion_matrix[key]) for key in confusion_matrix.keys() }


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

def print_stats(counts):
    sensitivity = calc_sensitivity(counts)
    specificity = calc_specificity(counts)
    ppv = calc_ppv(counts)
    f1 = calc_f1(counts)
    score = float(counts["TP"] + counts["TN"])/(counts["TP"] + counts["FP"] + counts["TN"] + counts["FN"] * 5)

    print("counts:  {}".format(counts))
    print("sensitivity:  {}".format(sensitivity))
    print("specificity:  {}".format(specificity))
    print("ppv:  {}".format(ppv))
    print("f1:  {}".format(f1))
    print("score:  {}".format(score))


# ## Run pipeline

def run(data_path, ann_path, filename, ecg_ann_type):
    print("ecg_ann_type:  {}".format(ecg_ann_type))
    print(" ann_path:  {}".format(ann_path))

    start = datetime.now()
    matrix = generate_confusion_matrix_dir(data_path, ann_path, ecg_ann_type)
    print("confusion matrix:  {}".format(matrix))
    print("total time:  {}".format(datetime.now() - start))

    with open(filename, "w") as f:
        json.dump(matrix, f)

def read_json(filename):
    with open(filename, "r") as f:
        dictionary = json.load(f)

    return dictionary

# print(datetime.now())
# write_filename = "sample_data/pipeline_fpinvalids_vtachfpann_nancheck.json"
# ecg_ann_type = "gqrs"
# run(data_path, ann_path, write_filename, ecg_ann_type)


if __name__ == '__main__':
    run(data_path, ann_path, write_filename, ecg_ann_type)

    matrix = read_json(write_filename)
    counts = get_counts(matrix)
    print_stats(counts)

    # matrix = read_json("../sample_data/baseline_performance/vtach_gqrs.json")
    # counts = get_counts(matrix)
    # print_stats(counts)

    # fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
    # print("missed true positives:  {}".format(get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TP")))
    # print("missed true negatives:  {}".format(get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TN")))


    # print("\nFP")
    # fp_matrix = read_json("sample_data/pipeline_fp.json")
    # counts_fp = get_counts(fp_matrix)
    # evaluate.print_stats(counts_fp)
    # print_by_type(fp_matrix['FN'])


    # print("\nFP invalids with GQRS")
    # fpinvalids_matrix = read_json("sample_data/pipeline_fpinvalids.json")
    # counts_fpinvalids = get_counts(fpinvalids_matrix)
    # evaluate.print_stats(counts_fpinvalids)
    # print_by_type(fpinvalids_matrix['FN'])

    # missed_true_negatives = get_missed(fpinvalids_matrix, fplesinger_confusion_matrix, "TN")
    # print("missed true positives:  {}".format(get_missed(fpinvalids_matrix, fplesinger_confusion_matrix, "TP")))
    # print("missed true negatives:  {}".format(missed_true_negatives))
    # print_by_type(missed_true_negatives)
    # print(len(missed_true_negatives))


    # print("\nFP invalids with GQRS without abp test in vtach")
    # fpinvalids_without_vtach_abp = read_json("sample_data/pipeline_fpinvalids_novtachabp.json")
    # counts_fpinvalids_without_vtach_abp = get_counts(fpinvalids_without_vtach_abp)
    # evaluate.print_stats(counts_fpinvalids_without_vtach_abp)
    # print_by_type(fpinvalids_without_vtach_abp['FN'])

#     print_by_type(gqrs_matrix['FN'])
#     print_by_arrhythmia(confusion_matrix_gqrs, 'v')

#     fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
#     print("missed true positives:  {}".format(get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TP")))
#     print("missed true negatives:  {}".format(get_missed(gqrs_matrix, fplesinger_confusion_matrix, "TN")))


# ## Comparing classification with other algorithms

# In[21]:

# def generate_others_confusion_matrices(filename, data_path):
#     others_confusion_matrices = {}

#     with open(filename, "r") as f:
#         reader = csv.DictReader(f)
#         authors = reader.fieldnames[1:]
#         for author in authors:
#             others_confusion_matrices[author] = { "TP": [], "FP": [], "FN": [], "TN": [] }

#         for line in reader:
#             sample_name = line['record name']
#             true_alarm = is_true_alarm(data_path, sample_name)

#             for author in authors:
#                 classified_true_alarm = line[author] == '1'
#                 matrix_classification = get_confusion_matrix_classification(true_alarm, classified_true_alarm)

#                 others_confusion_matrices[author][matrix_classification].append(sample_name)

#     return others_confusion_matrices


# # filename = "sample_data/answers.csv"
# # others_confusion_matrices = generate_others_confusion_matrices(filename, data_path)


# # In[7]:

# # for author in others_confusion_matrices.keys():
# #     other_confusion_matrix = others_confusion_matrices[author]
# #     print(author)
# #     counts = get_counts(other_confusion_matrix)
# #     evaluate.print_stats(counts)
# #     print_by_type(other_confusion_matrix['FN'])


# # In[23]:

# def get_missed(confusion_matrix, other_confusion_matrix, classification):
#     missed = []

#     for sample in other_confusion_matrix[classification]:
#         if sample not in confusion_matrix[classification]:
#             missed.append(sample)

#     return missed

# fplesinger_confusion_matrix = others_confusion_matrices['fplesinger-210']
# print("missed true positives:  {}".format(get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TP")))
# print("missed true negatives:  {}".format(get_missed(confusion_matrix_gqrs, fplesinger_confusion_matrix, "TN")))


# In[ ]:
