
# coding: utf-8

# # QRS detection

# In[16]:

import invalid_sample_detection   as invalid
import matplotlib.pyplot          as plt
import numpy                      as np
import parameters
import wfdb
import socket

# get_ipython().magic(u'matplotlib inline')

# determine the paths for data/annotations based off the computer name
hostname=socket.gethostname()

# if hostname=='alistair-pc70':
#     data_path = '/data/challenge-2015/data/'
#     ann_path = '/data/challenge-2015/ann/'
# else:
data_path = '../sample_data/challenge_training_data/'
ann_path = '../sample_data/challenge_training_multiann/'
fp_ann_path = '../sample_data/fplesinger_data/'


# ## Helper methods

# In[24]:

# Get annotation file type based on channel type and index
def get_ann_type(channel, channel_index, ecg_ann_type): 
    channel_type = invalid.get_channel_type(channel)
    if channel_type == "Resp": 
        return ""
    
    if ecg_ann_type == "fp": 
        return ann_type_fplesinger(channel_index)
    
    else: 
        return ann_type_qrs(channel_type, channel_index, ecg_ann_type)
    
    
# Get annotation file type for fplesinger ann files
def ann_type_fplesinger(channel_index): 
    return "fp" + str(channel_index)
    
    
# Get annotation file type for non-fplesinger ann files
def ann_type_qrs(channel_type, channel_index, ecg_ann_type): 
    if channel_type == "ECG": 
        ann_type = ecg_ann_type + str(channel_index)
    elif channel_type == "BP": 
        ann_type = 'wabp'
    elif channel_type == "PLETH": 
        ann_type = "wpleth"
    elif channel_type == "Resp": 
        ann_type = ""
    else: 
        raise Exception("Unrecognized ann type")
    
    return ann_type


# In[4]:

# # Start and end in seconds
# def get_annotation_annfs(sample, ann_type, start, end, channel_type): 
#     # Check annotation fs with guess of smaller val (DEFAULT_OTHER_FS) to prevent checking out of range data
#     ann_fs = parameters.DEFAULT_OTHER_FS

#     # Find annotation fs from wfdb.rdann
#     annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)

#     # If rdann's provided ann_fs is valid, use that annotation fs
#     if annotation[-1] is not None and annotation[-1] != 0 and isinstance(annotation[-1], (int, float)):  
#         ann_fs = annotation[-1]

#     # Otherwise, use default annotation fs and print warning if the annotation is not empty
#     elif len(annotation[0]) > 0: 
#         if channel_type == "ECG" or ann_type.startswith("fp"): 
#             ann_fs = parameters.DEFAULT_ECG_FS
#         else: 
#             print "Annotation fs defaulted to ", parameters.DEFAULT_OTHER_FS, " for ", sample, ann_type

#     # Get proper range of annotation based on annotation fs, only run if different ann_fs from before 
#     if ann_fs != parameters.DEFAULT_OTHER_FS: 
#         annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
                        
#     return annotation, ann_fs


# In[28]:

# Start and end in seconds
def get_annotation_annfs(sample, ann_type, start, end, channel_type): 
    if channel_type == "ECG" or ann_type.startswith("fp"): 
        ann_fs = parameters.DEFAULT_ECG_FS
    else: 
        ann_fs = parameters.DEFAULT_OTHER_FS

    # Find annotation fs from wfdb.rdann
    annotation = wfdb.rdann(sample, ann_type, sampfrom=int(start * ann_fs), sampto=int(end*ann_fs))

    # # If rdann's provided ann_fs is valid, use that annotation fs
    # if annotation[-1] is not None and annotation[-1] != 0 and isinstance(annotation[-1], (int, float)):  
    #     fs = annotation[-1]
    # else: 
    #     fs = ann_fs
        
    # # Get proper range of annotation based on annotation fs, only run if different ann_fs from before 
    # if fs != ann_fs: 
    #     ann_fs = fs
    #     annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
                        
    return annotation, ann_fs


# In[5]:

def get_ann_fs(channel_type, ecg_ann_type): 
    if channel_type == "ECG" or ecg_ann_type.startswith("fp"): 
        return parameters.DEFAULT_ECG_FS
    return parameters.DEFAULT_OTHER_FS


# In[6]:

# start and end in seconds
def get_annotation(sample, ann_type, ann_fs, start, end): 
    try: 
        annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
    except Exception as e: 
        annotation = []
        print(e)
    
    return annotation


# ## Calculating RR intervals

# In[7]:

# # Calculate RR intervals in the sample, where start and end in seconds
# def calculate_rr_intervals(sample, ann_type, start, end, channel_type): 
#     annotation, ann_fs = get_annotation_annfs(sample, ann_type, start, end, channel_type)
    
#     # Convert annotations sample numbers into seconds if >0 annotations in signal
#     if len(annotation[0]) > 0: 
#         ann_seconds = annotation[0] / float(ann_fs)
#     else: 
#         return np.array([0.0])
    
#     rr_intervals = np.array([])
#     for index in range(1, np.size(ann_seconds)):
#         rr_intervals = np.append(rr_intervals, round(ann_seconds[index] - ann_seconds[index - 1], 4))

#     return rr_intervals

# sample_name = "b220s"
# start, end = 284, 300
# rr_intervals = calculate_rr_intervals(ann_path + sample_name, 'jqrs1', start, end, "V")
# if len(rr_intervals) > 0: 
#     print "average: ", sum(rr_intervals) / len(rr_intervals)
# print "rr_intervals", rr_intervals


# In[29]:

def get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type, start=None, end=None):
    if start is None or end is None: 
        # Start and end given in seconds
        start, end, alarm_duration = invalid.get_start_and_end(fields)

    channels = fields['signame']
    channel = channels[channel_index]
    channel_type = invalid.get_channel_type(channel)
    channel_rr_intervals = np.array([])

    ann_type = get_ann_type(channel, channel_index, ecg_ann_type)
    try: 
        annotation, ann_fs = get_annotation_annfs(ann_path + sample_name, ann_type, start, end, channel_type)

        # Convert annotations sample numbers into seconds if >0 annotations in signal
        if len(annotation.annsamp) > 0: 
            ann_seconds = np.array(annotation[0]) / float(ann_fs)
        else: 
            return np.array([0.0])

        for index in range(1, np.size(ann_seconds)):
            channel_rr_intervals = np.append(channel_rr_intervals, round(ann_seconds[index] - ann_seconds[index - 1], 4))

    except Exception as e: 
        print(e)

    return channel_rr_intervals

# sample_name = "a103l"
# sig, fields = wfdb.rdsamp(data_path + sample_name)
# ecg_ann_type = "fp"
# channel_index = 2
# print get_channel_rr_intervals(fp_ann_path, sample_name, channel_index, fields, ecg_ann_type)


# In[9]:

# Start and end given in seconds
def get_rr_dict(ann_path, sample_name, fields, ecg_ann_type, start=None, end=None): 
    rr_dict = {}
    if start is None or end is None: 
        # Start and end given in seconds
        start, end, alarm_duration = invalid.get_start_and_end(fields)    
    
    channels = fields['signame']
    for channel_index in range(len(channels)): 
        channel_name = channels[channel_index]
        channel_type = invalid.get_channel_type(channel_name)
        if channel_type == "Resp": 
            continue
        
        rr_intervals = get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type, start, end)
        
        rr_dict[channel_name] = rr_intervals
    
    return rr_dict


# ## Plotting

# In[10]:

# Plot signal together with annotation types on the channel for data ranging from start to end
# start and end given in seconds
def plot_annotations(data_path, ann_path, fp_ann_path, sample_name, channel_index, start, end, ecg_ann_types, data_fs=250.0, loc=1): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channel_name = fields['signame'][channel_index]
    channel_type = invalid.get_channel_type(channel_name)
    time_vector = np.linspace(start, end, (end-start) * data_fs)
        
    # Plot the time series of the signal
    plt.figure(figsize=[8,5])
    plt.plot(time_vector, sig[int(start * data_fs):int(end * data_fs), channel_index], '-',
             color=parameters.COLORS[0], linewidth=2, 
             label=fields['signame'][channel_index])
    
    # Plot each annotation type
    for index in range(len(ecg_ann_types)):
        ecg_ann_type = ecg_ann_types[int(index)]
        ann_type = get_ann_type(channel_name, channel_index, ecg_ann_type)

        if ecg_ann_type == "fp": 
            path = fp_ann_path
        else: 
            path = ann_path

        annotation, ann_fs = get_annotation_annfs(path + sample_name, ann_type, start, end, channel_type)
        if len(annotation.annsamp) == 0: 
            plt.show()
            return

        annotation_seconds = annotation.annsamp / float(ann_fs)
        annotation_y = [ sig[int(ann_time * data_fs), channel_index] for ann_time in annotation_seconds ]
        plt.plot(annotation_seconds, annotation_y,
             color=parameters.COLORS[index+1],
             linestyle='none', linewidth=3,
             marker=parameters.MARKER_TYPES[0], markersize=9,
             label=ann_type)
            


    plt.xlabel('Time (seconds)',fontsize=12)
    plt.legend(fontsize=12, loc=loc)
    plt.grid()
    plt.show()

    return annotation

# In[18]:

data_fs = 250
sample_name = 'v736s'
start = 270
end = 280
# ecg_ann_type = ["gqrs", "jqrs", "fp"]
ecg_ann_type = ['fp']

# choose the lead to plot (annotations are generated off the first lead)
channel_index = 0

# plot_annotations(data_path, ann_path, fp_ann_path, sample_name, channel_index, start, end, ecg_ann_type, data_fs, loc=4)


# In[ ]:



