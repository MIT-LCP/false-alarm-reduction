

def abs_value_dist_metric(x, y):
    return abs(x-y)

def is_true_alarm(fields):
    return fields['comments'][1] == 'True alarm'

def get_alarm_classification(fields):
    return fields['comments'][0]

def get_channel_type(channel_name):
    channel_types_dict = {}
    with open("../data/sigtypes", "r") as f:
        for line in f:
            splitted_line = line.split("\t")
            channel = splitted_line[-1].rstrip()
            channel_type = splitted_line[0]
            channel_types_dict[channel] = channel_type

    if channel_name in channel_types_dict.keys():
        return channel_types_dict[channel_name]

    raise Exception("Unknown channel name")
