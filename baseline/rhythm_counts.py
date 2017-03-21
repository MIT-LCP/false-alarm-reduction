from parameters import *
import os
import wfdb


def read_signals(data_path):
    fields_dict = {}

    for filename in os.listdir(data_path):
        if filename.endswith(HEADER_EXTENSION):
            sample_name = filename.rstrip(HEADER_EXTENSION)

            sig, fields = wfdb.rdsamp(data_path + sample_name)

            fields_dict[sample_name] = fields

    return fields_dict

def get_num_pos_neg(fields_dict): 
	counts_by_arrhythmia = {}

	for sample_name in fields_dict.keys(): 
		fields = fields_dict[sample_name]
		arrhythmia = sample_name[0]

		if arrhythmia not in counts_by_arrhythmia.keys(): 
			counts_by_arrhythmia[arrhythmia] = [0,0] # num pos, num neg

		is_true_alarm = fields['comments'][1] == 'True alarm'
		if is_true_alarm: 
			counts_by_arrhythmia[arrhythmia][0] += 1
		else: 
			counts_by_arrhythmia[arrhythmia][1] += 1

	return counts_by_arrhythmia


if __name__ == '__main__': 
	data_path = "../../sample_data/challenge_training_data/"
	arrhythmias = ['a', 'b', 't', 'v', 'f']
	fields_dict = read_signals(data_path)

	counts_by_arrhythmia = get_num_pos_neg(fields_dict)
	print counts_by_arrhythmia