
# coding: utf-8

# In[ ]:

import invalid_sample_detection    as invalid
import matplotlib.pyplot           as plt
import numpy                       as np
import parameters
import load_annotations
import wfdb

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'config IPCompleter.greedy=True;')


# In[2]:

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_ann/'
sample_name = 'a170s'

invalids = invalid.calculate_invalids_standard(data_path + sample_name)
print invalid.calculate_cval(invalids)


# In[21]:

fs = parameters.FS
ann_type = 'jqrs'
start = 0
end = 5000
channel = 0
col = np.asarray(
    [[228,26,28],
    [55,126,184],
    [77,175,74],
    [152,78,163],
    [255,127,0],
    [255,255,51],
    [166,86,40],
    [247,129,191]])/256.0


sig, fields = wfdb.rdsamp(data_path + sample_name)
t = np.linspace(float(start)/fs,float(end)/fs,end-start)
plt.figure(figsize=[16,10])
plt.plot(t, sig[start:end,channel], '-',
         color=col[1],linewidth=2,
         label=fields['signame'][channel])
plt.show()

rr_intervals = calculate_rr_intervals_standard(ann_path + sample_name, channel, ann_type, start, end)
print sum(rr_intervals)/ len(rr_intervals)


# In[ ]:



