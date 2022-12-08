"""
The interface to load log datasets. The datasets currently supported include HDFS data.

Authors:
    LogPAI Team

"""


import pandas as pd
import numpy as np
from datetime import timedelta
import os, os.path
from fnmatch import fnmatch
from collections import OrderedDict
from datetime import datetime as dt
from dateutil.parser import parse
import torch
from torch.utils.data import TensorDataset


dataset_list = ['E9000刀片服务器下电',
                'E9000刀片服务器重启',
                'E9000服务器交换板下电',
                'E9000服务器交换板重启',
                'E9000服务器电源板故障',
                '交换机Eth-trunk端口LACP故障',
                '交换机端口频繁Up_Down',
                '存储系统管理链路故障']


def load_NAIE(log_file, interval):
    struct_log = pd.read_csv(log_file,
                             engine='c',
                             na_filter=False,
                             memory_map=True)
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        try:
            dt_str = row['Date'] + 'T' + row['Time']
        except:
            dt_str = row['DateTime']
        ts = parse(dt_str).timestamp()
        slice_win = ts // interval
        if slice_win not in data_dict:
            data_dict[slice_win] = []
        data_dict[slice_win].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()),
                           columns=['slice_win', 'EventSequence'])
    x_data = data_df['EventSequence'].values
    t_data = list(data_df['slice_win'].values)

    return x_data, t_data

def log_time_sequence(path, interval):
    log_file = [name for name in os.listdir(path)
                if fnmatch(name, '*.log_structured.csv')]
    log_file.sort()
    log_dict = OrderedDict()
    time_dict = OrderedDict()
    for file in log_file:
        name = file.split('.')[0]
        (x_train, t_data) = load_NAIE(os.path.join(path, file), interval)
        if name not in log_dict:
            log_dict[name] = []
        log_dict[name].append(list(x_train))
        time_dict[name] = t_data

    return log_dict, time_dict


def sequence_train(path_train, interval, window_size):
    # 1 log_dict
    log_dict, _ = log_time_sequence(path_train, interval)

    # 2 vocab2idx
    template_file = [name for name in os.listdir(path_train)
                           if fnmatch(name, '*.log_templates.csv')]
    template_file.sort()
    vocab2idx = {'PAD': 0}
    for file in template_file:
        template = pd.read_csv(os.path.join(path_train, file),
                               engine='c',
                               na_filter=False,
                               memory_map=True)
        for idx, template_id in enumerate(template['EventId'],
                                          start=len(vocab2idx)):
            vocab2idx[template_id] = idx
    vocab2idx['UNK'] = len(vocab2idx)

    # 3 train_loader
    num_windows = 0
    input = []
    output = []
    for name, seqs in log_dict.items():
        for line in seqs[0]:
            num_windows += 1
            if len(line) == 1:
                line += line
            if len(line) <= window_size:
                line = line[0:-1] + ['PAD'] * (window_size + 1 - len(line)) + [line[-1]]
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK']) for ID in line])
            for i in range(len(line) - window_size):
                input.append(line[i:i + window_size])
                output.append(line[i + window_size])
    print('Number of windows (train): {}'.format(num_windows))
    print('Number of instances (train): {}'.format(len(input)))
    train_loader = TensorDataset(torch.tensor(input, dtype=torch.float),
                                 torch.tensor(output))

    return train_loader, vocab2idx

def sequence_test(path_test, interval, vocab2idx, window_size):
    # 1 log_dict & time_dict
    log_dict, time_dict = log_time_sequence(path_test, interval)

    # 2 log_dict
    num_windows = 0
    for name, seqs in log_dict.items():
        log = []
        for line in seqs[0]:
            if len(line) == 1:
                line += line
            if len(line) <= window_size:
                line = line[0:-1] + ['PAD'] * (window_size + 1 - len(line)) + [line[-1]]
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK']) for ID in line])
            log.append(line)
        log_dict[name] = log
        num_windows += len(log)
    print('Number of windows (test):{}'.format(num_windows))

    return log_dict, time_dict


def log_count(x, vocab2idx):
    num_seq = len(x)
    num_event = len(vocab2idx)
    x_count = np.zeros((num_seq, num_event))
    for seq_idx in range(num_seq):
        for col in range(len(x[seq_idx])):
            x_count[seq_idx, x[seq_idx][col]] += 1

    return x_count

def log_frequency_train(x, setting):
    # 1 tf-idf
    if setting['use_tf_idf']:
        tf = np.sum(x > 0, axis=0)
        tf_idf = np.log(x.shape[0] / (tf + 1e-8))
        x = x * np.tile(tf_idf, (x.shape[0], 1))
    else:
        tf_idf = None

    # 2 zero-mean
    if setting['use_zero_mean']:
        mean = x.mean(axis=0)
        mean = mean.reshape(1, x.shape[1])
        x = x - np.tile(mean, (x.shape[0], 1))
    else:
        mean = None

    return x, tf_idf, mean

def log_frequency_test(x, setting, tf_idf, mean):
    # 1 tf-idf
    if setting['use_tf_idf'] and tf_idf is not None:
        x = x * np.tile(tf_idf, (x.shape[0], 1))

    # 2 sero-mean
    if setting['use_zero_mean'] and mean is not None:
        x = x - np.tile(mean, (x.shape[0], 1))

    return x

def frequency_train(path_train, interval, setting):
    # 1 log_dict
    log_dict, _ = log_time_sequence(path_train, interval)

    # 2 vocab2idx
    vocab2idx = {}
    template_file = [name for name in os.listdir(path_train)
                     if fnmatch(name, '*.log_templates.csv')]
    template_file.sort()
    for file in template_file:
        template = pd.read_csv(os.path.join(path_train, file),
                               engine='c',
                               na_filter=False,
                               memory_map=True)
        for idx, template_id in enumerate(template['EventId'], start=len(vocab2idx)):
            vocab2idx[template_id] = idx
    vocab2idx['UNK'] = len(vocab2idx)

    # 3 log_list
    log_list = []
    for name, seqs in log_dict.items():
        for line in seqs[0]:
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK'])
                          for ID in line])
            log_list.append(line)
    print('Number of instances (train): {}'.format(len(log_list)))

    # 4 count_list
    count_list = log_count(log_list, vocab2idx)

    # 5 frequency_list
    frequency_list, tf_idf, mean = log_frequency_train(count_list, setting)

    return frequency_list, vocab2idx, tf_idf, mean

def frequency_test(path_test, interval, vocab2idx, setting, tf_idf, mean):
    # 1 log_dict & time_dict
    log_dict, time_dict = log_time_sequence(path_test, interval)

    # 2 log_dict
    num_seqs = 0
    for name, seqs in log_dict.items():
        dataset = []
        for line in seqs[0]:
            line = tuple([vocab2idx.get(ID, vocab2idx['UNK'])
                          for ID in line])
            dataset.append(line)
        log_dict[name] = dataset
        num_seqs += len(dataset)
    print('Number of instances (test): {}'.format(num_seqs))

    # 3 frequency_dict
    frequency_dict = OrderedDict()
    for name, seqs in log_dict.items():
        count_list = log_count(seqs, vocab2idx)
        frequency_dict[name] = log_frequency_test(count_list, setting, tf_idf, mean)

    return frequency_dict, time_dict


def save_result_to_csv(y, t, result_dir, interval, timezone):
    # 1 generate result
    dataset = []
    datetime = []
    label = []
    for name, slices in t.items():
        for i, slice_win in enumerate(slices):
            time_slice = (dt.utcfromtimestamp(slice_win * interval)
                          + timedelta(hours=timezone)).strftime('%Y-%m-%d %H:%M:%S')
            dataset.append(name)
            datetime.append(time_slice)
            label.append(int(y[name][i]))
    data_dict = {'dataset': dataset, 'time_slice(UTC+8)': datetime, 'label': label}
    data_df = pd.DataFrame(data_dict)

    # 2 save result
    submit_df = pd.DataFrame()
    for name in dataset_list:
        df = data_df[data_df['dataset'] == name]
        submit_df = submit_df.append(df, ignore_index=True)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if os.path.exists(result_dir + 'submit.csv'):
        os.remove(result_dir + 'submit.csv')
    submit_df.to_csv(result_dir + 'submit.csv', index=False, encoding='utf-8')