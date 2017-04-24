# from ventricular_beat_detection import ventricular_beat_annotations_dtw
import scipy.signal         as scipy_signal
import scipy.fftpack        as scipy_fftpack
import matplotlib.pyplot    as plt
import numpy                as np
import parameters
import csv
import wfdb


##############################
##### Invalids detection #####
##############################

def band_pass_filter(signal, f_low, f_high, order, fs): 
    window = scipy_signal.firwin(order+1, [f_low, f_high], nyq=np.floor(fs/2), pass_zero=False,
                  window='hamming', scale=False)
    A = scipy_fftpack.fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(scipy_fftpack.fftshift(A / abs(A).max())))

    if np.size(signal) < 153: 
        return
    return scipy_signal.filtfilt(window, 1, signal)


def get_signal_fft(signal, signal_duration, fs): 
    # Number of samplepoints
    N = signal_duration * fs
    # sample spacing
    T = 1.0 / fs
    
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    signal_fft = scipy_fftpack.fft(signal)
    
    return (xf, 2.0/N * np.abs(signal_fft[:int(N/2)]))


# Check if amplitude within invalid range is above acceptable amplitudes
def is_amplitude_within_cutoff(signal, f_low, f_high, cutoff, order, fs):  
    filtered_signal = band_pass_filter(signal, f_low, f_high, order, fs)
    if filtered_signal is not None: 
        # Return False if any value in the filtered_signal is greater than cutoff
        return not (filtered_signal > cutoff).any()
    return True


# Check signal statistics to be within minimum and maximum values
def check_stats_within_cutoff(signal, channel_type, stats_cutoffs): 
    signal_min = np.amin(signal)
    signal_max = np.amax(signal)
    var_range = signal_max - signal_min
    channel_stats_cutoffs = stats_cutoffs[channel_type]
    
    # Check minimum and maximum signal values
    if signal_min < channel_stats_cutoffs["val_min"] or signal_max > channel_stats_cutoffs["val_max"]: 
        return False
    
    # Check signal range in value
    if var_range > channel_stats_cutoffs["var_range_max"] or var_range < channel_stats_cutoffs["var_range_min"]: 
        return False
    
    return True


# Check if signal contains NaN values
def contains_nan(signal): 
    return np.isnan(signal).any()


# Check borders between histogram buckets so the difference is within a cutoff value
def histogram_test(signal, histogram_cutoff): 
    top_percentile = np.percentile(signal, parameters.TOP_PERCENTILE)
    bottom_percentile = np.percentile(signal, parameters.BOTTOM_PERCENTILE)
    
    # Filter out top and bottom 1% for data on which to generate histogram
    adjusted_signal = signal[(signal >= bottom_percentile) & (signal <= top_percentile)]
    
    # Generate histogram with 10 buckets by default
    histogram = np.histogram(adjusted_signal)[0]
    
    # Calculate frequency diffs between neighboring buckets and return True if all diffs within cutoff
    diffs = np.diff(histogram)
    return not (diffs > histogram_cutoff).any()


# TODO: fix path for sigtypes file
def get_channel_type(channel_name): 
    channel_types_dict = {}
    with open("../sample_data/sigtypes", "r") as f: 
        for line in f: 
            splitted_line = line.split("\t")
            channel = splitted_line[-1].rstrip()
            channel_type = splitted_line[0]
            channel_types_dict[channel] = channel_type
    
    if channel_name in channel_types_dict.keys(): 
        return channel_types_dict[channel_name]
    
    raise Exception("Unknown channel name")
        
    
# Return list of channel indices for channels of type channel_type
def get_channels_of_type(channels, channel_type): 
    channel_indices = np.array([])
    
    for channel_index in range(len(channels)): 
        channel_name = channels[channel_index]
        if channel_type == get_channel_type(channel_name): 
            channel_indices = np.append(channel_indices, channel_index)
    
    return channel_indices


# Get start and end points (in seconds) to check depending on type of alarm signaled
def get_start_and_end(fields): 
    alarm_type = fields['comments'][0]
    if alarm_type not in parameters.TESTED_BLOCK_LENGTHS: 
        raise Exception("Unrecognized alarm type")

    tested_block_length = parameters.TESTED_BLOCK_LENGTHS[alarm_type]
    
    end = parameters.ALARM_TIME # in seconds, alarm always sounded at 300th second
    start = end - tested_block_length # in seconds
    
    return (start, end, tested_block_length)

# Returns whether signal is valid or not
def is_valid(signal, channel_type, f_low, f_high, histogram_cutoff, freq_amplitude_cutoff, stats_cutoffs, order, fs): 
    if channel_type == "Resp": 
        return True
    
    # Checks which return True if passing the test, False if not
    histogram_check = histogram_test(signal, histogram_cutoff)
    stats_check = check_stats_within_cutoff(signal, channel_type, stats_cutoffs)
    nan_check = not contains_nan(signal)
    checks = np.array([histogram_check, stats_check, nan_check])
    
    # If ECG signal, also check signal amplitude in frequency range within limits
    if channel_type == "ECG": 
        signal_amplitude_check = is_amplitude_within_cutoff(signal, f_low, f_high, freq_amplitude_cutoff, order, fs)
        checks = np.append(checks, signal_amplitude_check)
    
    return all(checks)


# Return invalids list given sig for a single channel
def calculate_channel_invalids(channel_sig,
                               channel_type,
                               fs=parameters.DEFAULT_ECG_FS,
                               block_length=parameters.BLOCK_LENGTH, 
                               order=parameters.ORDER,
                               f_low=parameters.F_LOW,
                               f_high=parameters.F_HIGH,
                               hist_cutoff=parameters.HIST_CUTOFF,
                               ampl_cutoff=parameters.AMPL_CUTOFF,
                               stats_cutoffs=parameters.STATS_CUTOFFS): 
    invalids = np.array([])
    start = 0 # in sample number
    
    # Check validity of signal for each block_length-long block
    while start < len(channel_sig): 
        signal = channel_sig[int(start):int(start + block_length*fs)]
        start += (block_length * fs)

        is_data_valid = is_valid(signal, channel_type, f_low, f_high, hist_cutoff, ampl_cutoff, stats_cutoffs, order, fs)
        
        if is_data_valid: 
            invalids = np.append(invalids, 0)
        else: 
            invalids = np.append(invalids, 1)
    
    return invalids

    
# Returns invalids dictionary mapping each channel to an invalids array representing validity of 0.8 second blocks
# Takes in sig and fields after already reading the sample file
def calculate_invalids_sig(sig, fields,
                           start=None,
                           end=None,
                           block_length=parameters.BLOCK_LENGTH, 
                           order=parameters.ORDER,
                           f_low=parameters.F_LOW,
                           f_high=parameters.F_HIGH,
                           hist_cutoff=parameters.HIST_CUTOFF,
                           ampl_cutoff=parameters.AMPL_CUTOFF,
                           stats_cutoffs=parameters.STATS_CUTOFFS): 
    
    channels = fields['signame']
    fs = fields['fs']
    if start is None or end is None: 
        start, end, alarm_duration = get_start_and_end(fields)
    window_start, window_end = start * fs, end * fs # in sample number
    
    invalids = {}
    
    # Generate invalids array for each channel 
    for channel_num in range(len(channels)): 
        start = window_start
        channel_name = channels[channel_num]
        channel_type = get_channel_type(channel_name)
        channel_sig = sig[:,channel_num]
        
        invalids_array = calculate_channel_invalids(channel_sig, channel_type)
        invalids[channel_name] = invalids_array
                 
    return invalids


# Calculate overall c_val of invalids list for a single channel (0 = invalid, 1 = valid)
def calculate_cval_channel(channel_invalids): 
    if len(channel_invalids) > 0: 
        return 1 - float(sum(channel_invalids)) / len(channel_invalids)
    return None


#######################
##### Annotations #####
#######################

# Get annotation file type based on channel type and index
def get_ann_type(channel, channel_index, ecg_ann_type): 
    channel_type = get_channel_type(channel)
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


def get_ann_fs(channel_type, ecg_ann_type): 
    if channel_type == "ECG" or ecg_ann_type.startswith("fp"): 
        return parameters.DEFAULT_ECG_FS
    return parameters.DEFAULT_OTHER_FS


# start and end in seconds
def get_annotation(sample, ann_type, ann_fs, start, end): 
    try: 
        annotation = wfdb.rdann(sample, ann_type, sampfrom=start*ann_fs, sampto=end*ann_fs)
    except Exception as e: 
        annotation = wfdb.Annotation(sample, ann_type, [], [])
        print("Error getting annotation for sample ", sample, e)
    
    return annotation


# start and end in seconds
def get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type, start=None, end=None):
    if start is None or end is None: 
        # Start and end given in seconds
        start, end, alarm_duration = get_start_and_end(fields)

    channels = fields['signame']
    channel = channels[channel_index]
    channel_type = get_channel_type(channel)
    channel_rr_intervals = np.array([])

    ann_type = get_ann_type(channel, channel_index, ecg_ann_type)

    try: 
        ann_fs = get_ann_fs(channel_type, ecg_ann_type)
        annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, start, end)

        # Convert annotations sample numbers into seconds if >0 annotations in signal
        if len(annotation.annsamp) > 0:
            ann_seconds = np.array(annotation.annsamp) / float(ann_fs)
        else: 
            return np.array([0.0])

        for index in range(1, np.size(ann_seconds)):
            channel_rr_intervals = np.append(channel_rr_intervals, round(ann_seconds[index] - ann_seconds[index - 1], 4))

    except Exception as e: 
        print("Error getting channel RR intervals for sample", sample_name, e)

    return channel_rr_intervals


# Start and end given in seconds
def get_rr_dict(ann_path, sample_name, fields, ecg_ann_type, start=None, end=None): 
    rr_dict = {}
    if start is None or end is None: 
        # Start and end given in seconds
        start, end, alarm_duration = get_start_and_end(fields)    
    
    channels = fields['signame']
    for channel_index in range(len(channels)): 
        channel_name = channels[channel_index]
        channel_type = get_channel_type(channel_name)
        if channel_type == "Resp": 
            continue
        
        rr_intervals = get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type, start, end)
        
        rr_dict[channel_name] = rr_intervals
    
    return rr_dict


############################
##### Regular activity #####
############################

# Check if standard deviation of RR intervals of signal are within limits
def check_rr_stdev(rr_intervals): 
    stdev = np.std(rr_intervals)
            
    if stdev > parameters.RR_STDEV: 
        return False
    return True

# Check if heart rate, calculated by number of RR intervals in signal, are within limits
def check_heart_rate(rr_intervals, alarm_duration): 
    hr = (len(rr_intervals) + 1.) / alarm_duration * parameters.NUM_SECS_IN_MIN
            
    if hr > parameters.HR_MAX or hr < parameters.HR_MIN: 
        return False
    return True

# Check if sum of RR intervals is within limit of total duration, to ensure beats are evenly spaced throughout
def check_sum_rr_intervals(rr_intervals, alarm_duration): 
    min_sum = alarm_duration - parameters.RR_MIN_SUM_DIFF
        
    rr_sum = sum(rr_intervals)
        
    if rr_sum < min_sum: 
        return False
    return True    

# Check if total number of RR intervals is greater than a minimum 
def check_num_rr_intervals(rr_intervals): 
    if len(rr_intervals) < parameters.MIN_NUM_RR_INTERVALS: 
        return False
    return True


# Returns False if any block within signal is identified as invalid (invalid sample detection)
def check_invalids(invalids, channel): 
    if channel not in invalids.keys(): 
        raise Exception("Unknown channel")
    
    block_invalids_sum = sum(invalids[channel])
    if block_invalids_sum > parameters.INVALIDS_SUM: 
        return False
    return True

# Returns True for a given channel if all regular activity tests checked pass
def check_interval_regular_activity(rr_intervals, invalids, alarm_duration, channel): 
    all_checks = np.array([])
    
    # If the RR intervals should be checked but all annotations missing, auto fail
    if len(rr_intervals) == 0: 
        return False
    
    stdev_check = check_rr_stdev(rr_intervals)
    hr_check = check_heart_rate(rr_intervals, alarm_duration)
    sum_check = check_sum_rr_intervals(rr_intervals, alarm_duration)
    num_check = check_num_rr_intervals(rr_intervals)
    invalids_check = check_invalids(invalids, channel)
    
    all_checks = np.append(all_checks, [stdev_check, hr_check, sum_check, num_check, invalids_check])
                
    return np.all(all_checks)


# Determines regular activity of sample based on RR intervals and invalids array: 
# param: rr_dict as a dictionary of the form: 
#         { channel0: [rr_intervals], channel1: [rr_intervals], ...}
# param: alarm_duration duration of alarm in seconds
def is_rr_invalids_regular(rr_dict, invalids, alarm_duration, arrhythmia_type,
                           should_check_invalids=True, should_check_rr=True, should_num_check=True): 

    for channel in rr_dict.keys(): 
        channel_type = get_channel_type(channel)
        
        if arrhythmia_type == "Ventricular_Flutter_Fib" and channel_type != "ECG": 
            continue       
        
        rr_intervals = rr_dict[channel]
        is_regular = check_interval_regular_activity(rr_intervals, invalids, alarm_duration, channel)
        
        # If any channel is regular, reject alarm as false alarm
        if is_regular: 
            return True
    return False 


# Check overall sample for regular activity by iterating through each channel.
# If any channel exhibits regular activity, alarm indicated as false alarm.
def is_sample_regular(data_path, 
                      ann_path, 
                      sample_name, 
                      ecg_ann_type,
                      start=None, 
                      end=None,
                      verbose=False): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']
    nonresp_channels = [ channels.index(channel) for channel in channels if channel != "RESP" ]
    
    if start is None or end is None: 
        start, end, alarm_duration = get_start_and_end(fields)
    else: 
        alarm_duration = end - start
    
    # try: 
    #     invalids = {}
    #     for channel_index in nonresp_channels: 
    #         channel = channels[channel_index]

    #         with open(fp_ann_path + sample_name + "-invalids.csv", "r") as f: 
    #             reader = csv.reader(f)
    #             channel_invalids = [ int(float(row[channel_index])) for row in reader]
    #             invalids[channel] = channel_invalids[start*250:end*250]
                        
    # except Exception as e: 
    #     print("Error finding invalids for sample " + sample_name, e)
    #     invalids = calculate_invalids_sig(sig, fields, start, end)

    invalids = calculate_invalids_sig(sig, fields, start, end)

    for channel_index in range(len(channels)): 
        channel = channels[channel_index]
        channel_type = get_channel_type(channel)
        
        # Ignore respiratory channel
        if channel_type == "Resp": 
            continue
           
        alarm_prefix = sample_name[0]
        # Only use ECG channels for ventricular fib
        if alarm_prefix == "f": 
            if channel_type != "ECG": 
                continue
            
        rr = get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type)
            
        is_regular = check_interval_regular_activity(rr, invalids, alarm_duration, channel)
                
        # If any channel exhibits regular activity, deem signal as regular activity
        if is_regular: 
            return True
    return False



################################
##### Specific arrhythmias #####
################################

def test_asystole(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']
    fs = fields['fs']
        
    # Start and end given in seconds
    start, end, alarm_duration = get_start_and_end(fields)
    current_start = start
    current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    
    max_score = 0
    
    while current_end < end:
        start_index, end_index = int(current_start*fs), int(current_end*fs) 
        subsig = sig[start_index:end_index,:]
        summed_asystole_score = calc_summed_asystole_score(ann_path, sample_name, subsig, channels, ecg_ann_type,
                                                           current_start, current_end, verbose)
        max_score = max(max_score, summed_asystole_score)
        
        current_start += parameters.ASYSTOLE_ROLLING_INCREMENT
        current_end = current_start + parameters.ASYSTOLE_WINDOW_SIZE
    
    if verbose: 
        print(sample_name + " has max asystole score: " + str(max_score))
    
    return max_score > 0


def test_bradycardia(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']

    # Start and end given in seconds
    start, end, alarm_duration = get_start_and_end(fields)
    
    rr_intervals_list = get_rr_intervals_list(ann_path, sample_name, ecg_ann_type, fields, start, end)    
    best_channel_rr = find_best_channel(rr_intervals_list, alarm_duration)
    min_hr = get_min_hr(best_channel_rr, parameters.BRADYCARDIA_NUM_BEATS)
    
    if verbose: 
        print(sample_name + " with min HR: " + str(min_hr))
    
    return min_hr < parameters.BRADYCARDIA_HR_MIN


def test_tachycardia(data_path, ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']
        
    # Start and end given in s#econds
    start, end, alarm_duration = get_start_and_end(fields)

    rr_intervals_list = get_rr_intervals_list(ann_path, sample_name, ecg_ann_type, fields, start, end)        
    if check_tachycardia_channel(rr_intervals_list, alarm_duration): 
        return True
    
    best_channel_rr = find_best_channel(rr_intervals_list, alarm_duration)    
    max_hr = get_max_hr(best_channel_rr, parameters.TACHYCARDIA_NUM_BEATS)
    
    if verbose: 
        print(sample_name + " with max HR: " + str(max_hr))
    
    return max_hr > parameters.TACHYCARDIA_HR_MAX


def test_ventricular_tachycardia(data_path, 
                                 ann_path, 
                                 sample_name, 
                                 ecg_ann_type,
                                 verbose=False,
                                 fs=parameters.DEFAULT_ECG_FS, 
                                 order=parameters.ORDER, 
                                 num_beats=parameters.VTACH_NUM_BEATS,
                                 std_threshold=parameters.VTACH_ABP_THRESHOLD,
                                 window_size=parameters.VTACH_WINDOW_SIZE,
                                 rolling_increment=parameters.VTACH_ROLLING_INCREMENT): 

    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']
        
    # Start and end given in seconds
    start_time, end_time, alarm_duration = get_start_and_end(fields)
    alarm_sig = sig[int(start_time*fs):int(end_time*fs),:]
    
    ecg_channels = get_channels_of_type(channels, "ECG")
    abp_channels = get_channels_of_type(channels, "BP")
    
    # Initialize R vector
    size = int((alarm_duration - window_size) / rolling_increment) + 1
    r_vector = [0.] * size
    
    # Adjust R vector based on ventricular beats in signal
    for channel_index in ecg_channels:
        index = int(channel_index)
        ann_type = get_ann_type(channels[index], index, ecg_ann_type)
            
        r_delta = get_ventricular_beats_scores(alarm_sig[:,int(index)], ann_path, sample_name, ann_type, start_time, end_time)
        r_vector = r_vector + r_delta
                
        if verbose: 
            channel_sig = alarm_sig[:,index]
            lf, sub = get_lf_sub(channel_sig, order)
            ventricular_beats = ventricular_beat_annotations(lf, sub, ann_path + sample_name, ann_type, start_time, end_time, verbose)
            max_hr = max_ventricular_hr(ventricular_beats, num_beats, fs)
            print(str(sample_name) + " on channel "  + str(channels[int(channel_index)]) + " with max ventricular HR: ", str(max_hr))
          
    return any([ r_value > 0 for r_value in r_vector ])


def test_ventricular_flutter_fibrillation(data_path, 
                                          ann_path, 
                                          sample_name, 
                                          ecg_ann_type,
                                          verbose=False,
                                          fs=parameters.DEFAULT_ECG_FS,
                                          ann_fs=parameters.DEFAULT_ECG_FS,
                                          std_threshold=parameters.VFIB_ABP_THRESHOLD,
                                          window_size=parameters.VFIB_WINDOW_SIZE,
                                          rolling_increment=parameters.VFIB_ROLLING_INCREMENT):
    sig, fields = wfdb.srdsamp(data_path + sample_name)
    channels = fields['signame']
        
    # Start and end given in seconds
    start, end, alarm_duration = get_start_and_end(fields)
    alarm_sig = sig[int(start*fs):int(end*fs),:]

    ecg_channels = get_channels_of_type(channels, "ECG")
    abp_channels = get_channels_of_type(channels, "BP")
    
    # Find max duration of low frequency signal from all channels
    dlfmax = 0
    for channel_index in ecg_channels: 
        channel_index = int(channel_index)
        channel_dlfmax = calculate_dlfmax(alarm_sig[:,channel_index])
        dlfmax = max(dlfmax, channel_dlfmax)
                    
    # Initialize R vector to a value based on the D_lfmax (duration of low frequency)
    if dlfmax > parameters.VFIB_DLFMAX_LIMIT: 
        r_vector_value = 1.
    else: 
        r_vector_value = 0.
    size = int((alarm_duration - window_size) / rolling_increment) + 1
    r_vector = [r_vector_value] * size
        
    # Adjust R vector based on whether standard deviation of ABP channel is > or < the threshold
    for channel_index in abp_channels: 
        r_delta = get_abp_std_scores(alarm_sig[:,int(channel_index)], std_threshold, window_size, rolling_increment)
        r_vector = r_vector + r_delta
            
    # Adjust R vector based on dominant frequency in signal
    for channel_index in ecg_channels: 
        channel_index = int(channel_index)

        dominant_freqs = get_dominant_freq_array(alarm_sig[:,channel_index])
        regular_activity = get_regular_activity_array(alarm_sig, fields, ann_path, sample_name, ecg_ann_type)
        adjusted_dominant_freqs = adjust_dominant_freqs(dominant_freqs, regular_activity)
        
        new_r_vector = np.array([])
        for dominant_freq, r_value in zip(adjusted_dominant_freqs, r_vector): 
            if dominant_freq < parameters.VFIB_DOMINANT_FREQ_THRESHOLD: 
                new_r_vector = np.append(new_r_vector, 0.)
            else: 
                new_r_vector = np.append(new_r_vector, r_value)
        
        r_vector = new_r_vector
    
    return any([ r_value > 0 for r_value in r_vector ])


###############################################
##### Specific arrhythmias - helper funcs #####
###############################################

def calc_summed_asystole_score(ann_path, 
                               sample_name, 
                               subsig, 
                               channels, 
                               ecg_ann_type, 
                               current_start, 
                               current_end,
                               verbose=False, 
                               data_fs=parameters.DEFAULT_FS): 
    summed_score = 0
    
    for channel_index, channel in zip(range(len(channels)), channels): 
        channel_type = get_channel_type(channel)
        if channel_type == "Resp": 
            continue
        
        channel_subsig = subsig[:,channel_index]
        invalids = calculate_channel_invalids(channel_subsig, channel_type)
        cval = calculate_cval_channel(invalids)
        
        ann_type = get_ann_type(channel, channel_index, ecg_ann_type)
        ann_fs = get_ann_fs(channel_type, ecg_ann_type)
        
        annotation = get_annotation(ann_path + sample_name, ann_type, ann_fs, current_start, current_end)
           
        if len(annotation.annsamp) > 0: 
            current_score = -cval
        else: 
            current_score = cval        
        
        if verbose: 
            plt.figure(figsize=[7,5])
            plt.plot(channel_subsig, 'g-')
            annotation_seconds = annotation.annsamp / float(ann_fs)
            ann_x = [ (seconds - current_start) * data_fs for seconds in annotation_seconds ]
            ann_y = [ channel_subsig[index] for index in ann_x ]
            plt.plot(ann_x, ann_y, 'bo', markersize=8)
            plt.show()

            print(sample_name + ": " + channel + " [" + str(current_start) + ", " + str(current_end) + "] " + str(current_score))
        
        summed_score += current_score
        
    return summed_score   


def get_rr_intervals_list(ann_path, sample_name, ecg_ann_type, fields, start, end): 
    channels = fields['signame']

    rr_intervals_list = []
    
    for channel_index in range(len(channels)): 
        channel_name = channels[channel_index]
        channel_type = get_channel_type(channel_name)
        if channel_type == "Resp": 
            continue
            
        rr_intervals = get_channel_rr_intervals(ann_path, sample_name, channel_index, fields, ecg_ann_type, start, end)
        rr_intervals_list.append(rr_intervals)
        
    return rr_intervals_list


# Precondition: len(rr_intervals_list) > 0
# Return RR intervals with the min stdev of all the RR intervals in the list
def min_stdev_rr_intervals(rr_intervals_list): 
    opt_rr_intervals = []
    min_stdev = float('inf')
    
    for rr_intervals in rr_intervals_list:  
        stdev = np.std(rr_intervals)
        if stdev < min_stdev: 
            opt_rr_intervals = rr_intervals
            min_stdev = stdev
    
    return opt_rr_intervals


# Best channel: minimum stdev with acceptable RR intervals sum and count
# If none with acceptable RR interval sum and count --> select minimum stdev out of all RR intervals
def find_best_channel(rr_intervals_list, alarm_duration): 
    count_and_sum = []
    only_one_test = []
    for rr_intervals in rr_intervals_list: 
        sum_check = check_sum_rr_intervals(rr_intervals, alarm_duration)
        num_check = check_num_rr_intervals(rr_intervals)
        
        if sum_check and num_check: 
            count_and_sum.append(rr_intervals)
        
        elif sum_check or num_check: 
            only_one_test.append(rr_intervals)

    if len(count_and_sum) > 0: 
        return min_stdev_rr_intervals(count_and_sum)
    
    if len(only_one_test) > 0: 
        return min_stdev_rr_intervals(only_one_test)
    
    return min_stdev_rr_intervals(rr_intervals_list)            


def get_min_hr(rr_intervals, num_beats_per_block): 
    min_hr = float('inf')
    
    for index in range(num_beats_per_block, len(rr_intervals)): 
        subsection = rr_intervals[index - num_beats_per_block:index]
        hr = float(num_beats_per_block) / sum(subsection) * parameters.NUM_SECS_IN_MIN
        min_hr = min(min_hr, hr)
        
    return min_hr


def check_tachycardia_channel(rr_intervals_list, alarm_duration): 
    for rr_intervals in rr_intervals_list: 
        stdev_check = check_rr_stdev(rr_intervals)
        sum_check = check_sum_rr_intervals(rr_intervals, alarm_duration)
        hr_check = check_heart_rate(rr_intervals, alarm_duration)
        if stdev_check and sum_check and not hr_check:
            return True
            
    return False


def get_max_hr(rr_intervals, num_beats_per_block): 
    max_hr = -float('inf')
    
    for index in range(num_beats_per_block, len(rr_intervals)): 
        subsection = rr_intervals[index - num_beats_per_block:index]
        hr = float(num_beats_per_block) / sum(subsection) * parameters.NUM_SECS_IN_MIN
        max_hr = max(max_hr, hr)
        
    return max_hr


def hilbert_transform(x, fs, f_low, f_high, demod=False):
    N = len(x)
    f = scipy_fftpack.fft(x, n=N)
    i_high = int(np.floor(float(f_high)/fs*N))
    i_low = int(np.floor(float(f_low)/fs*N))
    win = scipy_signal.hamming( i_high - i_low )
    
    f[0:i_low] = 0
    f[i_low:i_high] = f[i_low:i_high]*win
    f[i_high+1:] = 0
    
    if demod==True:
        # demodulate the signal, i.e. shift the freq spectrum to 0
        i_mid = int( np.floor((i_high+i_low)/2.0) )
        f = np.concatenate( [f[i_mid:i_high], np.zeros(len(f)-(i_high-i_low)), f[i_low:i_mid] ]  )
        
    return 2*np.abs(scipy_fftpack.ifft(f, n=N))


def get_lf_sub(channel_sig, order): 
    lf = abs(hilbert_transform(channel_sig, parameters.DEFAULT_ECG_FS, parameters.LF_LOW, parameters.LF_HIGH))
    mf = abs(hilbert_transform(channel_sig, parameters.DEFAULT_ECG_FS, parameters.MF_LOW, parameters.MF_HIGH))
    hf = abs(hilbert_transform(channel_sig, parameters.DEFAULT_ECG_FS, parameters.HF_LOW, parameters.HF_HIGH))
    sub = mf - hf
    
    return lf, sub


# Return list of ventricular beats for ECG channels
def ventricular_beat_annotations(lf_subsig, sub_subsig, sample, ann_type, start_time, end_time, 
                                 verbose=True,
                                 fs=parameters.DEFAULT_FS,
                                 threshold_ratio=parameters.VENTRICULAR_BEAT_THRESHOLD_RATIO,
                                 ann_fs=parameters.DEFAULT_ECG_FS):    
    annotation = get_annotation(sample, ann_type, ann_fs, start_time, end_time)
        
    single_peak_indices = [ index - ann_fs * start_time for index in annotation.annsamp ]
        
    ventricular_beat_indices = np.array([])
    nonventricular_beat_indices = np.array([])
    
    for index in single_peak_indices:
        if index >= len(lf_subsig) or index >= len(sub_subsig): 
            continue
        
        index = int(index)
        if lf_subsig[index] > sub_subsig[index]: 
            ventricular_beat_indices = np.append(ventricular_beat_indices, index)
        else: 
            nonventricular_beat_indices = np.append(nonventricular_beat_indices, index)

    print(ventricular_beat_indices)

    if verbose: 
        plt.figure(figsize=[8,5])
        plt.plot(sub_subsig,'b-')
        plt.plot(lf_subsig,'r-')
        plt.plot(nonventricular_beat_indices, [sub_subsig[int(index)] for index in nonventricular_beat_indices], 'bo', markersize=8)
        plt.plot(ventricular_beat_indices, [ lf_subsig[int(index)] for index in ventricular_beat_indices ], 'ro', markersize=8)
        plt.show()
    
    return ventricular_beat_indices


def max_ventricular_hr(ventricular_beats, num_beats, fs):     
    max_hr = 0
    
    if len(ventricular_beats) < num_beats: 
        return max_hr
    
    for index in range(num_beats-1, len(ventricular_beats)): 
        sublist = ventricular_beats[index-num_beats+1:index]
        start_time = ventricular_beats[index-num_beats+1] / fs
        end_time = ventricular_beats[index] / fs

        hr = (num_beats-1) / (end_time - start_time) * parameters.NUM_SECS_IN_MIN         
        max_hr = max(hr, max_hr)    
        
    return max_hr

def read_ventricular_beat_annotations(sample_name, output_path="../sample_name/vtach_beat_ann/"): 
    ventricular_beats = []
    nonventricular_beats = []

    with open(output_path + sample_name + ".csv", 'r') as f: 
        reader = csv.DictReader(f)

        for row in reader: 
            if row['is_true_beat'] == '1': 
                ventricular_beats.append(int(row['ann_index']))
            else: 
                nonventricular_beats.append(int(row['ann_index']))

    return ventricular_beats, nonventricular_beats


def get_ventricular_beats_scores(channel_sig,
                                 ann_path, 
                                 sample_name,
                                 ann_type,
                                 initial_start_time, 
                                 initial_end_time,
                                 fs=parameters.DEFAULT_FS,
                                 order=parameters.ORDER, 
                                 max_hr_threshold=parameters.VTACH_MAX_HR,
                                 num_beats=parameters.VTACH_NUM_BEATS,
                                 window_size=parameters.VTACH_WINDOW_SIZE,
                                 rolling_increment=parameters.VTACH_ROLLING_INCREMENT):
    r_delta = np.array([])
    end = window_size * fs
    
    lf, sub = get_lf_sub(channel_sig, order)
    
    while end <= channel_sig.size: 
        start = end - window_size * fs
        start_index, end_index = int(start), int(end)

        channel_subsig = channel_sig[start_index:end_index]
        lf_subsig = lf[start_index:end_index]
        sub_subsig = sub[start_index:end_index]
        start_time = initial_start_time + start/fs
        end_time = start_time + window_size
        
        ventricular_beats = ventricular_beat_annotations(lf_subsig, sub_subsig, ann_path + sample_name, ann_type, start_time, end_time)
        # ventricular_beats, nonventricular_beats = read_ventricular_beat_annotations(sample_name)

        max_hr = max_ventricular_hr(ventricular_beats, num_beats, fs)
            
        invalids = calculate_channel_invalids(channel_subsig, "ECG")
        cval = calculate_cval_channel(invalids)
                
        if max_hr > max_hr_threshold: 
            r_delta = np.append(r_delta, cval)
        else: 
            r_delta = np.append(r_delta, 0) #-cval)
            
        end += (rolling_increment * fs)
            
    return r_delta


def get_abp_std_scores(channel_sig, 
                       std_threshold,
                       window_size,
                       rolling_increment,
                       fs=parameters.DEFAULT_FS):
    r_delta = np.array([])
    end = window_size * fs
    
    while end <= channel_sig.size: 
        start = end - window_size * fs
        start_index, end_index = int(start), int(end)

        channel_subsig = channel_sig[start_index:end_index]
        end += (rolling_increment * fs)

        invalids = calculate_channel_invalids(channel_subsig, "BP")
        cval = calculate_cval_channel(invalids)
                
        std = np.std(channel_subsig)        
        if std > std_threshold: 
            r_delta = np.append(r_delta, 0) #-cval)
        else: 
            r_delta = np.append(r_delta, cval)
            
    return r_delta


def calculate_dlfmax(channel_sig, 
                     order=parameters.ORDER): 
    lf, sub = get_lf_sub(channel_sig, order)
    
    current_dlfmax_start = None
    dlfmax_duration = 0
    prev_low_dominance = 0
    
    for index in range(len(lf)): 
        lf_sample = lf[index]
        sub_sample = sub[index]
        
        if lf_sample > sub_sample: 
            # If not yet started a low dominance area, set the start index
            if current_dlfmax_start is None: 
                current_dlfmax_start = index
                            
            # If a separate low dominance area, reset
            elif index - prev_low_dominance > parameters.VFIB_LOW_DOMINANCE_INDEX_THRESHOLD: 
                # Calculate duration of previous low dominance area and update max dlfmax
                duration = prev_low_dominance - current_dlfmax_start
                dlfmax_duration = max(dlfmax_duration, duration)
                
                # Start new area of low dominance
                current_dlfmax_start = index
            
            # Update previous index seen with low frequency dominance
            prev_low_dominance = index
          
        # Handle duration at the end of the segment
        if current_dlfmax_start is not None: 
            duration = prev_low_dominance - current_dlfmax_start
            dlfmax_duration = max(dlfmax_duration, duration)
                        
    return dlfmax_duration
            

# Get dominant freq in signal in rolling window
def get_dominant_freq_array(channel_sig, 
                            fs=parameters.DEFAULT_ECG_FS,
                            window_size=parameters.VFIB_WINDOW_SIZE,
                            rolling_increment=parameters.VFIB_ROLLING_INCREMENT): 
    
    end = window_size * fs
    dominant_freqs = np.array([])
    
    while end < channel_sig.size: 
        start = end - window_size * fs

        start_index, end_index = int(start), int(end)
        channel_subsig = channel_sig[start_index:end_index]
        end += (rolling_increment * fs)
        
        xf, fft = get_signal_fft(channel_subsig, window_size, fs)
        
        # Index of the fft is 2 * the actual frequency 
        dominant_freq = np.argmax(fft) / 2
        
        dominant_freqs = np.append(dominant_freqs, dominant_freq)
    return dominant_freqs


def get_regular_activity_array(sig,
                               fields,
                               ann_path, 
                               sample_name,
                               ecg_ann_type,
                               arrhythmia_type="Ventricular_Flutter_Fib",
                               fs=parameters.DEFAULT_ECG_FS,
                               window_size=parameters.VFIB_WINDOW_SIZE,
                               rolling_increment=parameters.VFIB_ROLLING_INCREMENT): 
    regular_activity_array = np.array([])
    end = window_size * fs
    
    while end < sig[:,0].size: 
        start = end - window_size * fs
        start_index, end_index = int(start), int(end)
        subsig = sig[start_index:end_index]
        
        invalids_dict = calculate_invalids_sig(subsig, fields)
        rr_dict = get_rr_dict(ann_path, sample_name, fields, ecg_ann_type, start/fs, end/fs)
                
        is_regular = is_rr_invalids_regular(rr_dict, invalids_dict, window_size, arrhythmia_type)
        if is_regular: 
            regular_activity_array = np.append(regular_activity_array, 1)
        else: 
            regular_activity_array = np.append(regular_activity_array, 0)

        end += (rolling_increment * fs)
    
    return regular_activity_array


def adjust_dominant_freqs(dominant_freqs, regular_activity): 
    adjusted_dominant_freqs = np.array([])

    for freq, is_regular in zip(dominant_freqs, regular_activity): 
        if is_regular: 
            adjusted_dominant_freqs = np.append(adjusted_dominant_freqs, 0)
        else: 
            adjusted_dominant_freqs = np.append(adjusted_dominant_freqs, freq)
    
    return adjusted_dominant_freqs



####################
##### Pipeline #####
####################

# Returns true if alarm is classified as a true alarm
# Uses fplesinger annotations for ventricular tachycardia test
def classify_alarm(data_path, ann_path, fp_ann_path, sample_name, ecg_ann_type, verbose=False): 
    sig, fields = wfdb.srdsamp(data_path + sample_name)

    if ecg_ann_type == 'fp': 
        ann_path = fp_ann_path

    is_regular = is_sample_regular(data_path, ann_path, sample_name, ecg_ann_type)    
    if is_regular:
        return False

    alarm_type = sample_name[0]
    if alarm_type == "a": 
        arrhythmia_test = test_asystole

    elif alarm_type == "b": 
        arrhythmia_test = test_bradycardia
    
    elif alarm_type == "t": 
        arrhythmia_test = test_tachycardia
    
    elif alarm_type == "v": 
        ann_path = fp_ann_path
        ecg_ann_type = 'fp'
        arrhythmia_test = test_ventricular_tachycardia
    
    elif alarm_type == "f": 
        arrhythmia_test = test_ventricular_flutter_fibrillation
    
    else: 
        raise Exception("Unknown arrhythmia alarm type")
    
    # try: 
    return arrhythmia_test(data_path, ann_path, sample_name, ecg_ann_type, verbose)
    # except Exception as e: 
    #     print("sample_name: ", sample_name, e)
    #     return True


if __name__ == '__main__': 
    data_path = '../sample_data/challenge_training_data/'
    ann_path = '../sample_data/challenge_training_multiann/'
    fp_ann_path = '../sample_data/fplesinger_data/'
    ecg_ann_type = 'gqrs'

    print(classify_alarm(data_path, ann_path, fp_ann_path, "v199l", ecg_ann_type))
