
# coding: utf-8

# # QRS detection

# In[16]:

import matplotlib.pyplot    as plt
import numpy                as np
import parameters
import wfdb
import socket

get_ipython().magic(u'matplotlib inline')

# determine the paths for data/annotations based off the computer name
hostname=socket.gethostname()

if hostname=='alistair-pc70':
    data_path = '/data/challenge-2015/data/'
    ann_path = '/data/challenge-2015/ann/'
else:
    data_path = 'sample_data/challenge_training_data/'
    ann_path = 'sample_data/challenge_training_multiann/'


# In[65]:

# Plot signal together with annotation types on the channel for data ranging from start to end
def plot_annotations(data_path, ann_path, sample_name, ann_types_list, channel, data_fs, ann_fs, start, end): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    time_vector = np.linspace(start, end, (end-start) * data_fs)
        
    # Plot the time series of the signal
    plt.figure(figsize=[16, 10])
    plt.plot(time_vector, sig[start * data_fs:end * data_fs, channel], '-',
             color=parameters.COLORS[0], linewidth=2, 
             label=fields['signame'][channel])
    
    if len(ann_types_list) > len(parameters.MARKER_TYPES) or len(ann_types_list) > len(parameters.COLORS)-1: 
        raise RuntimeException("too many types of annotations to plot")
    
    # Plot each annotation type
    for index in range(len(ann_types_list)): 
        ann_type = ann_types_list[index]
        annotation = wfdb.rdann(ann_path + sample_name, ann_type, sampfrom = start * ann_fs, sampto = end * ann_fs)
        print "ann_type: ", ann_type
        print "annotation: ", annotation
        
        if len(annotation[0]) == 0: 
            plt.show()
            return
        
        annotation_seconds = annotation[0] / float(ann_fs)
        annotation_y = [ sig[int(ann_time * data_fs), channel] for ann_time in annotation_seconds ]
        plt.plot(annotation_seconds, annotation_y,
             color=parameters.COLORS[index + 1],
             linestyle='none', linewidth=3,
             marker=parameters.MARKER_TYPES[index], markersize=12,
             label=ann_type)
        
    plt.xlabel('Time (seconds)',fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.show()


# In[77]:

data_fs = 250
ann_fs = 125
sample_name = 'v131l'
start = 290
end = 300

# choose the lead to plot (annotations are generated off the first lead)
channel = 2

plot_annotations(data_path, ann_path, sample_name, ['wabp'], channel, data_fs, ann_fs, start, end)


# In[79]:

# Calculate RR intervals in the sample, where 
def calculate_rr_intervals(sample, ann_type, start, end): 
    dummy_ann = wfdb.rdann(sample, ann_type, sampfrom = start, sampto = end)
    ann_fs = dummy_ann[-1]
    if ann_fs is None or ann_fs == 0 or not isinstance(ann_fs, (int, float)): 
        ann_fs = parameters.DEFAULT_FS
        print "Annotation fs defaulted to ", parameters.DEFAULT_FS
    annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)

    # Convert annotations sample numbers into seconds if annotations in signal
    if len(annotation[0]) > 0: 
        ann_seconds = annotation[0] / float(ann_fs)
    else: 
        return [0.0]
    
    rr_intervals = np.array([])
    for index in range(1, len(ann_seconds)):
        rr_intervals = np.append(rr_intervals, round(ann_seconds[index] - ann_seconds[index - 1], 4))
        
    return rr_intervals

sample_name = "v131l"
start, end = 290, 300
rr_intervals = calculate_rr_intervals(ann_path + sample_name, 'wabp', start, end)
if len(rr_intervals) > 0: 
    print "average: ", sum(rr_intervals) / len(rr_intervals)
print "rr_intervals", rr_intervals
        


# In[ ]:



