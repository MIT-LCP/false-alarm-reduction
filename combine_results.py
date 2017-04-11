import csv
import json

path = "sample_data/"
baseline_filename = "pipeline_fpinvalids_vtachfpann.json"
dtw_filename = "dtw_fromfile.json"

with open(path + baseline_filename, 'r') as f:
    baseline_matrix = json.load(f) 

with open(path + dtw_filename, 'r') as f: 
    dtw_matrix = json.load(f)

with open(path + "answers.csv", 'w') as csvfile: 
    fieldnames = ['sample_name', 'baseline_is_classified_true', 'dtw_is_classified_true', 'is_true']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for classification in baseline_matrix: 
        if classification == "FP": 
            is_true = 0
            baseline_is_classified_true = 1
        
        elif classification == "TP": 
            is_true = 1
            baseline_is_classified_true = 1

        elif classification == "FN": 
            is_true = 1
            baseline_is_classified_true = 0
        
        else: # TN
            is_true = 0 
            baseline_is_classified_true = 0

        for sample_name in baseline_matrix[classification]: 
            if sample_name in dtw_matrix["TP"] or sample_name in dtw_matrix["FP"]: 
                dtw_is_classified_true = 1
            else: # "TN" or "FN"
                dtw_is_classified_true = 0

            row = {
                'sample_name': sample_name,
                'baseline_is_classified_true': baseline_is_classified_true,
                'dtw_is_classified_true': dtw_is_classified_true,
                'is_true': is_true
            }
            writer.writerow(row)

