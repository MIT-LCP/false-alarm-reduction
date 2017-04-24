from spectrum                   import *
from scipy.stats                import kurtosis
from sklearn.linear_model       import LogisticRegression
from datetime                   import datetime
import numpy                    as np
import matplotlib.pyplot  		as plt
import csv
import wfdb

data_path = "sample_data/challenge_training_data/"
answers_filename = "sample_data/answers.csv"

start_time = 290
end_time = 300
fs = 250.

TRAINING_THRESHOLD = 600


def get_psd(channel_subsig, order): 
    ar, rho, ref = arburg(channel_subsig, order)
    psd = arma2psd(ar, rho=rho, NFFT=512)
    psd = psd[len(psd):len(psd)/2:-1]

    # plt.figure()
    # plt.plot(10*np.log10(abs(psd)*2./(2.*np.pi)))
    # plt.title('PSD')
    # plt.ylabel('Log of PSD')
    # plt.xlabel('Frequency (Hz)')
    # plt.show()

    return psd


def get_baseline(subsig, ecg_channels, order=30): 
    channel_index = ecg_channels[0]
    channel_subsig = subsig[:,int(channel_index)]

    psd = get_psd(channel_subsig, order)

    numerator_min_freq = 1
    numerator_max_freq = 2
    denominator_min_freq = 1
    denominator_max_freq = 40

    numerator = sum(psd[numerator_min_freq:numerator_max_freq+1])
    denominator = sum(psd[denominator_min_freq:denominator_max_freq+1])

    baseline = float(numerator) / denominator
    return baseline


def get_power(subsig, ecg_channels, order=30):
    channel_index = ecg_channels[0]
    channel_subsig = subsig[:,int(channel_index)]

    psd = get_psd(channel_subsig, order)

    numerator_min_freq = 5
    numerator_max_freq = 15
    denominator_min_freq = 5
    denominator_max_freq = 40

    numerator = sum(psd[numerator_min_freq:numerator_max_freq+1])
    denominator = sum(psd[denominator_min_freq:denominator_max_freq+1])

    power = float(numerator) / denominator
    return power


def get_ksqi(subsig, ecg_channels):
    channel_index = ecg_channels[0]
    channel_subsig = subsig[:,int(channel_index)] 
	
    # TODO: this uses fisher as default (with normal of 0) versus pearson's (with normal of 3)
    ksqi = kurtosis(channel_subsig) - 3

    if abs(ksqi) >= 25: 
        return 25
    return ksqi


def get_pursqi(subsig, ecg_channels): 
    channel_index = ecg_channels[0]
    channel_subsig = subsig[:,int(channel_index)] 

    s = channel_subsig
    sd = np.diff(channel_subsig);
    sdd = np.zeros(len(channel_subsig))
    for i in range(len(channel_subsig)): 
        if i == 0: 
            sdd[i] = channel_subsig[i+1] - 2*channel_subsig[i]

        elif i == len(channel_subsig) - 1: 
            sdd[i] = 2*channel_subsig[i] + channel_subsig[i-1]

        else: 
            sdd[i] = channel_subsig[i+1] - 2*channel_subsig[i] + channel_subsig[i-1]

    w0 = (2*np.pi/len(s))*sum(np.square(s))   # 2pi E[s^2]=2pi Rs(0)
    w2 = (2*np.pi/len(s))*sum(np.square(sd))  # 2pi Ts^2 E[sd^2], 
    w4 = (2*np.pi/len(s))*sum(np.square(sdd)) #2pi Ts^4 E[sdd^2]

    pursqi = (w2**2)/(w0*w4)
    return pursqi


def get_channel_type(channel_name): 
    channel_types_dict = {}
    with open("sample_data/sigtypes", "r") as f: 
        for line in f: 
            splitted_line = line.split("\t")
            channel = splitted_line[-1].rstrip()
            channel_type = splitted_line[0]

            if channel_name == channel: 
            	return channel_type

    raise Exception("Unknown channel name")
        
    
# Return list of channel indices for channels of type channel_type
def get_channels_of_type(channels, channel_type): 
    channel_indices = np.array([])
    
    for channel_index in range(len(channels)): 
        channel_name = channels[channel_index]
        if channel_type == get_channel_type(channel_name): 
            channel_indices = np.append(channel_indices, channel_index)
    
    return channel_indices


# x includes sample names --> exclude for classification
# training = sample num < 600
# testing = sample num > 600
def generate_training_testing(): 
    training_x, training_y = [], []
    testing_x, testing_y = [], []

    with open(answers_filename, 'r') as f: 
        reader = csv.DictReader(f, fieldnames=['sample_name', 'baseline_is_classified_true', 'dtw_is_classified_true', 'is_true'])
        reader.next()

        for row in reader: 
            sample_name = row['sample_name']
            sample_number = sample_name[1:-1]

            sig, fields = wfdb.rdsamp(data_path + sample_name)
            subsig = sig[int(start_time*fs):int(end_time*fs),:]
            ecg_channels = get_channels_of_type(fields['signame'], "ECG")

            if len(ecg_channels) == 0: 
		    	print "NO ECG CHANNELS FOR SAMPLE: ", sample_name
		    	continue

            baseline = get_baseline(subsig, ecg_channels)
            power = get_power(subsig, ecg_channels)
            ksqi = get_ksqi(subsig, ecg_channels)
            pursqi = get_pursqi(subsig, ecg_channels)

            x_val = [ 
                row['sample_name'], 
                row['baseline_is_classified_true'], 
                row['dtw_is_classified_true'],
                baseline,
                power,
                ksqi, 
                pursqi
            ]

            if int(sample_number) < TRAINING_THRESHOLD: 
                training_x.append(x_val)
                training_y.append(row['is_true'])
            else: 
                testing_x.append(x_val)
                testing_y.append(row['is_true'])


    return training_x, training_y, testing_x, testing_y


start = datetime.now()
print "Starting at", start
print "Generating datasets..."
training_x, training_y, testing_x, testing_y = generate_training_testing()

print "Running classifier..."
classifier = LogisticRegression()
classifier.fit(training_x, training_y)
print classifier.score(testing_x, testing_y)

print datetime.now() - start
