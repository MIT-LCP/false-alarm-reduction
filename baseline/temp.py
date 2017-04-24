import csv
import wfdb
import matplotlib.pyplot as plt

vtach_beats = []
non_vtach_beats = []
sample_name = "v135l"

with open("../sample_data/vtach_beat_ann/" + sample_name + ".csv", 'r') as f: 
	reader = csv.DictReader(f)

	for row in reader: 
		if row['is_true_beat'] == '1': 
			vtach_beats.append(int(row['ann_index']))

		else: 
			non_vtach_beats.append(int(row['ann_index']))

sig, fields = wfdb.rdsamp("../sample_data/challenge_training_data/" + sample_name)

channel_subsig = sig[290*250:300*250,0]
vtach_indices = [ ann_index - 290*250 for ann_index in vtach_beats ]
print [ index / 250. for index in vtach_indices]
non_vtach_indices = [ ann_index - 290*250 for ann_index in non_vtach_beats ]

plt.figure()
plt.plot(channel_subsig, 'g-')
plt.plot(vtach_indices, [ channel_subsig[index] for index in vtach_indices ], 'ro')
plt.plot(non_vtach_indices, [ channel_subsig[index] for index in non_vtach_indices ], 'bo')
plt.show()

