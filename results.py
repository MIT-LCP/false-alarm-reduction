
# coding: utf-8

# # Summary of results

# In[30]:

import matplotlib.pyplot  as plt
import load_annotations   as annotate
import parameters
import pipeline
import wfdb

get_ipython().magic(u'matplotlib inline')

data_path = 'sample_data/challenge_training_data/'
ann_path = 'sample_data/challenge_training_multiann/'
ecg_ann_type = 'gqrs'
data_fs = parameters.DEFAULT_FS


# ## Overview of algorithm

# ![title](figures/overview.jpg)

# ##### Invalid sample detection: 
# Valid if passes all tests: 
# - Histogram check
# - Values within reasonable range
# - NaN check
# - Not too much noise in range 70-90 Hz
# 
# c_val gives ratio of valid data to overall signal (1.0 = completely valid data)
# 
# ##### Regular activity test: 
# Regular activity if passes all tests: 
# - Valid data (as given by invalid sample detection tests)
# - Standard deviation of RR intervals
# - Average HR over entire signal within reasonable limits
# - Sum of RR intervals is close enough to entire length of signal
# - Number of total RR intervals above threshold
# 
# ##### Asystole test: 
# 
# 
#     

# ## Overall comparison of results

# #### Confusion matrix for current algorithm: 
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 245                         | 97        |
# | **False** | <font color='red'>49</font> | 359       |
# 
# |           | **True**                        | **False** |        
# | ---       | :---:                           | :---:     |
# | **True**  | 0.327                           | 0.129     |
# | **False** | <font color='red'>0.0653</font> | 0.479       |
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 1        | 8         | 2         | 1                | 37        |

# #### Confusion matrix for other algorithms: 
# 
# ##### fplesinger-210: 
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 275                         | 64        |
# | **False** | <font color='red'>19</font> | 392       |
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 1        | 1         | 1         | 1                | 15        |
# 
# 
# ##### l.m.eerikainen-209:
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 294                         | 65        |
# | **False** | <font color='red'>0</font>  | 391       |
# 
# 
# ##### bellea-212:
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 291                         | 327       |
# | **False** | <font color='red'>3</font>  | 129       |
# 
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 3        | 0         | 0         | 0                | 0         |
# 
# ##### hoog.antink-216:
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 291                         | 89        |
# | **False** | <font color='red'>3</font>  | 367       |
# 
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 0        | 0         | 0         | 0                | 3         |
# 
# ##### vxk106120-213:
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 280                         | 74        |
# | **False** | <font color='red'>14</font> | 382       |
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 0        | 0         | 0         | 0                | 14        |
# 
# 
# ##### bestcly-204:
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 277                         | 133       |
# | **False** | <font color='red'>17</font> | 323       |
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 0        | 3         | 1         | 0                | 13        |
# 
# 
# ##### sibylle.fallet-210: 
# |           | **True**                    | **False** |        
# | ---       | :---:                       | :---:     |
# | **True**  | 276                         | 108       |
# | **False** | <font color='red'>18</font> | 348       |
# 
# | **Asys** | **Brady** | **Tachy** | **Vfib/flutter** | **Vtach** |
# | :---:    | :---:     | :---:     | :---:            | :---:     |
# | 0        | 1         | 4         | 0                | 13        |
# 

# ## Examples

# In[3]:

def classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    true_alarm = pipeline.is_true_alarm(data_path, sample_name)
    classified_true_alarm = pipeline.is_classified_true_alarm(data_path, ann_path, sample_name, ecg_ann_type, verbose)
    matrix_classification = pipeline.get_confusion_matrix_classification(true_alarm, classified_true_alarm)

    title = matrix_classification + ": " + sample_name
    plot_signal(data_path, sample_name, title)

    
def plot_signal(data_path, sample_name, plot_title=""): 
    sig, fields = wfdb.rdsamp(data_path + sample_name)
    fs = fields['fs']
    channels = fields['signame']
    non_resp_channels = [ index for index in range(len(channels)) if channels[index] != "RESP" ]
    alarm_type = fields['comments'][0]
    tested_block_length = parameters.TESTED_BLOCK_LENGTHS[alarm_type]
    
    start_time, end_time = parameters.ALARM_TIME - tested_block_length, parameters.ALARM_TIME
    start, end = int(start_time * fs), int(end_time * fs)
    wfdb.plotwfdb(sig[start:end, non_resp_channels], fields, title=plot_title)
    


# ### Asystole

# In[4]:

sample_name = "a161l"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type)


# In[5]:

sample_name = "a670s"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type)


# Annotations indicated for pacemarker artefacts despite not QRS --> asystole not detected because QRS complexes annotated in ECG channels

# In[6]:

channel_index = 0
start, end = 294, 298.5
annotate.plot_annotations(data_path, ann_path, sample_name, channel_index, start, end, ecg_ann_type, data_fs)


# ### Bradycardia

# In[4]:

sample_name = "b734s"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type, verbose=True)


# The cutoff HR for bradycardia is 45 bpm. Most of the false negatives have heart rates only slightly greater than the cutoff. 

# In[12]:

sample_name = "b187l"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type, verbose=True)


# Very different annotations between the different ECG channels. II channel was chosen by the algorithm to find the min HR in determining bradycardia.

# In[13]:

start, end = 284, 300
annotate.plot_annotations(data_path, ann_path, sample_name, 0, start, end, ecg_ann_type, data_fs)
annotate.plot_annotations(data_path, ann_path, sample_name, 1, start, end, ecg_ann_type, data_fs)


# ### Tachycardia

# In[25]:

sample_name = "t418s"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type, verbose=True)


# II channel was chosen by the algorithm to find the max HR in determining tachycardia. This ECG data segment has a high heart rate for a short period of time at the end of the segment right before the alarm gets triggered. Because a high heart rate is only seen at the end of the segment, the overall HR for a segment of at least 12 beats (as necessitated by the algorithm) is not seen to be higher than the minimum needed to trigger tachycardia.

# In[27]:

start, end = 290, 300
annotate.plot_annotations(data_path, ann_path, sample_name, 0, start, end, ecg_ann_type, data_fs, loc=2)
annotate.plot_annotations(data_path, ann_path, sample_name, 1, start, end, ecg_ann_type, data_fs, loc=3)


# In[28]:

sample_name = "t700s"
classify_and_plot_signal(data_path, ann_path, sample_name, ecg_ann_type, verbose=True)


# II channel is chosen by the algorithm as the channel by which to determine tachycardia, even though the annotations for the second channel (V) is much cleaner. This is because of the criterion used to determine the "best" channel: min standard deviation of all the channels that satisfy the other tests (# of annotations and sum of the annotations > min threshold). This is maybe not the best/most relevant criterion in deciding the best test, and we should likely explore something else to eliminate these false negatives. 

# In[29]:

start, end = 290, 300
annotate.plot_annotations(data_path, ann_path, sample_name, 0, start, end, ecg_ann_type, data_fs, loc=4)
annotate.plot_annotations(data_path, ann_path, sample_name, 1, start, end, ecg_ann_type, data_fs, loc=4)


# ### Ventricular tachycardia

# In[ ]:




# ## Obvious avenues of improvement

# 1. Improve criteria for selecting "best" channel for bradycardia and tachycardia
# 2. Fewer beats necessary to detect tachycardia 
# 3. Better annotations for QRS complexes
# 4. Ventricular tachycardia detected in one channel can be canceled out by vtach not detected in other channel

# In[ ]:



