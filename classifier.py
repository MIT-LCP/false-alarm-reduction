from spectrum                   import *
from scipy.stats                import kurtosis
from sklearn.linear_model       import LogisticRegression, LassoCV
from sklearn.metrics            import auc, roc_curve
from datetime                   import datetime
import numpy                    as np
import matplotlib.pyplot  		as plt
import csv
import wfdb

data_path = "sample_data/challenge_training_data/"
answers_filename = "sample_data/answers.csv"
features_filename = "sample_data/features.csv"

start_time = 290
end_time = 300
fs = 250.

TRAINING_THRESHOLD = 600


def get_psd(channel_subsig, order, nfft): 
    channel_subsig = channel_subsig-np.mean(channel_subsig)
    ar, rho, ref = arburg(channel_subsig, order)
    psd = arma2psd(ar, rho=rho, NFFT=nfft)
    psd = psd[len(psd):len(psd)/2:-1]

    # plt.figure()
    # plt.plot(linspace(0, 1, len(psd)), abs(psd)*2./(2.*np.pi))
    # plt.title('PSD')
    # plt.ylabel('Log of PSD')
    # plt.xlabel('Normalized Frequency')
    # plt.show()

    # print len(psd)

    return psd


def get_baseline(channel_subsig, order=30, nfft=1024): 
    psd = get_psd(channel_subsig, order, nfft)

    numerator_min_freq = int(0 * nfft / 125.)
    numerator_max_freq = int(1  * nfft / 125.)
    denominator_min_freq =int( 0  * nfft / 125.)
    denominator_max_freq = int(40 * nfft / 125.)

    numerator = sum(psd[numerator_min_freq:numerator_max_freq+1])
    denominator = sum(psd[denominator_min_freq:denominator_max_freq+1])

    baseline = float(numerator) / denominator
    return 1 - baseline


def get_power(channel_subsig, order=30, nfft=1024):
    psd = get_psd(channel_subsig, order, nfft)

    numerator_min_freq =int( 5  * nfft / 125.)
    numerator_max_freq = int(15 * nfft / 125.)
    denominator_min_freq = int(5 * nfft / 125.)
    denominator_max_freq = int(40 * nfft / 125.)

    numerator = sum(psd[numerator_min_freq:numerator_max_freq+1])
    denominator = sum(psd[denominator_min_freq:denominator_max_freq+1])

    power = float(numerator) / denominator
    return power


def get_ksqi(channel_subsig):
    # TODO: this uses fisher as default (with normal of 0) versus pearson's (with normal of 3)
    ksqi = kurtosis(channel_subsig) - 3

    if abs(ksqi) >= 25: 
        return 25
    return ksqi


def get_pursqi(channel_subsig): 
    s = channel_subsig
    sd = np.diff(channel_subsig);
    sdd = np.zeros(len(channel_subsig))
    for i in range(len(channel_subsig)): 
        if i == 0: 
            sdd[i] = channel_subsig[2] - 2*channel_subsig[1] + channel_subsig[0]

        elif i == len(channel_subsig) - 1: 
            sdd[i] = channel_subsig[-1] - 2*channel_subsig[-2] + channel_subsig[-3]

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
def generate_features(features_filename): 
    training_x, training_y = [], []
    testing_x, testing_y = [], []

    with open(features_filename, 'w') as fo: 
        writer = csv.writer(fo)
        writer.writerow(['sample_name', 'is_training', 'is_true', 'baseline', 'dtw', 'psd', 'power', 'ksqi', 'pursqi'])

        with open(answers_filename, 'r') as f: 
            reader = csv.reader(f)
            headers = reader.next()

            reader = csv.DictReader(f, fieldnames=headers)

            for row in reader: 
                sample_name = row['sample_name']
                sample_number = sample_name[1:-1]

                sig, fields = wfdb.rdsamp(data_path + sample_name)
                subsig = sig[int(start_time*fs):int(end_time*fs),:]
                ecg_channels = get_channels_of_type(fields['signame'], "ECG")

                if len(ecg_channels) == 0: 
    		    	print "NO ECG CHANNELS FOR SAMPLE: ", sample_name
    		    	continue

                channel_subsig = subsig[:,int(ecg_channels[0])]

                try: 
                    baseline = get_baseline(channel_subsig)
                    power = get_power(channel_subsig)
                    ksqi = get_ksqi(channel_subsig)
                    pursqi = get_pursqi(channel_subsig)

                except Exception as e: 
                    print "sample_name:", sample_name, e
                    continue

                if np.isnan([baseline, power, ksqi, pursqi]).any(): 
                    print "sample containing nan:", sample_name, [baseline, power, ksqi, pursqi]
                    continue

                if int(sample_number) < TRAINING_THRESHOLD: 
                    is_training = 1
                else: 
                    is_training = 0

                x_val = [ 
                    row['sample_name'], 
                    is_training,
                    int(row['is_true']),
                    int(row['baseline_is_classified_true']), 
                    int(row['dtw_is_classified_true']),
                    baseline,
                    power,
                    ksqi, 
                    pursqi
                ]

                writer.writerow(x_val)

def generate_datasets(features_filename): 
    training_x, training_y, testing_x, testing_y = [], [], [], []

    with open(features_filename, 'r') as f: 
        reader = csv.reader(f)
        headers = reader.next()

        reader = csv.DictReader(f, fieldnames=headers)

        for row in reader: 
            x_val = [
                int(row['baseline']),
                int(row['dtw']),
                float(row['psd']),
                float(row['power']),
                float(row['ksqi']),
                float(row['pursqi'])
            ]
            y_val = int(row['is_true'])

            if int(row['is_training']) == 1 and row['sample_name'][0] == 'v': 
                training_x.append(x_val)
                training_y.append(y_val)
            elif row['sample_name'][0] == 'v': 
                testing_x.append(x_val)
                testing_y.append(y_val)
    return training_x, training_y, testing_x, testing_y


def get_score(prediction, true):
    
    TP = np.sum([(prediction[i] == 1) and (true[i] == 1) for i in range(len(prediction))])
    TN = np.sum([(prediction[i] == 0) and (true[i] == 0) for i in range(len(prediction))])
    FP = np.sum([(prediction[i] == 1) and (true[i] == 0) for i in range(len(prediction))])
    FN = np.sum([(prediction[i] == 0) and (true[i] == 1) for i in range(len(prediction))])

    # print TP, TN, FP, FN

    numerator =  TP + TN
    denominator = FP + 5*FN + numerator

    return float(numerator) / denominator


# print "Generating datasets..."
# # generate_features(features_filename)
# training_x, training_y, testing_x, testing_y = generate_datasets(features_filename)

# print len(training_y), len(testing_y)


#     # start = datetime.now()
#     # print "Starting at", start
#     # print "Generating datasets..."
#     # training_x, training_y, testing_x, testing_y = generate_training_testing()



# print "Running classifier..."
# classifier = LogisticRegression(penalty='l1')
# # lasso = LassoCV()
# classifier.fit(training_x, training_y)

#     # probability of class 1 (versus 0)
#     # predictions_y = classifier.predict_proba(testing_x)[:,1]
#     # score = classifier.score(testing_x, testing_y)

#     # fpr, tpr, thresholds = roc_curve(testing_y, predictions_y)
#     # auc = auc(fpr, tpr)

#     # print "auc: ", auc
#     # print "score: ", score
#     # print "fpr: ", fpr, "tpr: ", tpr

#     # # plt.figure()
#     # # plt.title("ROC curve for DTW-only classiifer")
#     # # plt.xlabel("False positive rate")
#     # # plt.ylabel("True positive rate")
#     # # plt.plot(fpr, tpr)
#     # # plt.show()
# # lasso.fit(training_x, training_y)
# # predictions_y = lasso.predict(testing_x)

# fpr, tpr, thresholds = roc_curve(testing_y, predictions_y)

# chall_score = list()
# for th in thresholds:
#     chall_score.append(get_score([x >= th for x in predictions_y], testing_y))


# auc = auc(fpr, tpr)

# print classifier.coef_
# print "auc: ", auc
# print "score: ", score
# print "fpr: ", fpr, "tpr: ", tpr


# plt.figure()
# plt.title("ROC curve for top-level classifier with challenge scores")
# plt.xlabel("False positive rate")
# plt.ylabel("True positive rate")
# plt.plot(fpr, tpr, label='ROC Curve')
# plt.plot(fpr, chall_score, label='Challenge score')
# plt.show()

#     # DTW only
#     # auc:  0.461675144589
#     # score:  0.529166666667

#     # Baseline only 
#     # auc:  0.877012054909
#     # score:  0.875

#     # Combined
#     # auc:  0.910041112118
#     # score:  0.841666666667


