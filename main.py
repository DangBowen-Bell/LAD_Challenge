import os, os.path
from tqdm import tqdm
from collections import OrderedDict
from naie.datasets import get_data_reference
from naie.context import Context
from naie.log import logger
import moxing as mox

from logparser import Drain
from loglizer.models import DeepLog
from loglizer.models import PCA
from loglizer.models import InvariantsMiner
from loglizer.models import LogClustering
from loglizer import dataloader


path_train = './data/structured_logs/NAIE/train_data'
path_test = './data/structured_logs/NAIE/test_data'
result_dir = './result/'
timezone = 8

log_format_dict = {
    'CE交换机五天无故障日志数据集': '<DateTime> <<Pid>><Content>',
    'E9000服务器五天无故障日志数据集': '<DateTime> <Content>',
    'OceanStor存储五天无故障日志数据集': '<Invalid_Date> <<Pid>><Date> <Time> <IP> <Component> <Content>',
    'E9000刀片服务器下电': '<DateTime> <Content>',
    'E9000刀片服务器重启': '<DateTime> <Content>',
    'E9000服务器电源板故障': '<DateTime> <Content>',
    'E9000服务器交换板下电': '<DateTime> <Content>',
    'E9000服务器交换板重启': '<DateTime> <Content>',
    '存储系统管理链路故障': '<Invalid_Date> <<Pid>><Date> <Time> <IP> <Component> <Content>',
    '交换机Eth-trunk端口LACP故障': '<DateTime> <<Pid>><Content>',
    '交换机端口频繁Up_Down': '<DateTime> <<Pid>><Content>',
}
regex = [
    r'blk_(|-)[0-9]+',
    r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)',
    r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
]
st = 0.5
depth = 4

sequence_model_name = ['DeepLog']
frequency_model_name = ['PCA', 'InvariantsMiner', 'LogClustering']
default_model_cfg = \
    {'DeepLog':{'input_size': 1,
                'hidden_size': 64,
                'num_layers': 2,
                'window_size': 10,
                'batch_size': 1024,
                'num_epochs': 100,
                'lr': 2.01e-4,
                'topN': 10,
                'use_pretrained_model': False,
                'pretrained__deeplog_model_path': './result/DeepLog/baseline_deeplog_model_cuda.pt'},
     'PCA': {'use_tf_idf': True,
             'use_zero_mean': True,
             'n_components': 0.95,
             'c_alpha': 3.2905},
     'InvariantsMiner': {'use_tf_idf': False,
                         'use_zero_mean': False,
                         'percentage': 0.98,
                         'epsilon': 0.5,
                         'scale_list': [1, 2, 3, 4, 5]},
     'LogClustering': {'use_tf_idf': True,
                       'use_zero_mean': False,
                       'max_dist': 0.3,
                       'anomaly_threshold': 0.3,
                       'mode': 'online',
                       'num_bootstrap_samples': 1000}}
interval = 300

def parse_log():
    # 1 train data
    input_dir = '/cache/train_data/'
    output_dir = './data/structured_logs/NAIE/train_data'
    data_reference = get_data_reference(dataset="DatasetService",
                                        dataset_entity="log_abnormal_training_dataset")
    file_paths = data_reference.get_files_paths()
    mox.file.copy_parallel(os.path.dirname(file_paths[0]), input_dir)

    log_file_lst = os.listdir(input_dir)
    log_file_lst.sort()
    print(f"train_data: {log_file_lst}")
    for i in tqdm(range(len(log_file_lst))):
        fname, fext = os.path.splitext(log_file_lst[i])
        log_format = log_format_dict[fname]
        parser = Drain.LogParser(log_format,
                                 indir=input_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex)
        parser.parse(log_file_lst[i])

    # 2 test_data
    input_dir = '/cache/test_data'
    output_dir = './data/structured_logs/NAIE/test_data'
    data_reference = get_data_reference(dataset="DatasetService",
                                        dataset_entity="log_abnormal_test_dataset")
    file_paths = data_reference.get_files_paths()
    mox.file.copy_parallel(os.path.dirname(file_paths[0]), input_dir)

    log_file_lst = os.listdir(input_dir)
    log_file_lst.sort()
    print(f"test_data: {log_file_lst}")
    for i in tqdm(range(len(log_file_lst))):
        fname, fext = os.path.splitext(log_file_lst[i])
        log_format = log_format_dict[fname]
        parser = Drain.LogParser(log_format,
                                 indir=input_dir,
                                 outdir=output_dir,
                                 depth=depth,
                                 st=st,
                                 rex=regex)
        parser.parse(log_file_lst[i])

def sequence_model(model_name):
    # 1 load data
    xy_train, vocab2idx = dataloader.sequence_train(path_train, interval,
                                                    default_model_cfg['DeepLog']['window_size'])
    xy_test, time_dict = dataloader.sequence_test(path_test, interval, vocab2idx,
                                                  default_model_cfg['DeepLog']['window_size'])

    # 2 train
    if model_name == 'DeepLog':
        model = DeepLog(input_size=default_model_cfg['DeepLog']['input_size'],
                        hidden_size=default_model_cfg['DeepLog']['hidden_size'],
                        num_layers=default_model_cfg['DeepLog']['num_layers'],
                        num_classes=len(vocab2idx),
                        window_size=default_model_cfg['DeepLog']['window_size'],
                        batch_size=default_model_cfg['DeepLog']['batch_size'],
                        num_epochs=default_model_cfg['DeepLog']['num_epochs'],
                        lr=default_model_cfg['DeepLog']['lr'],
                        topN=default_model_cfg['DeepLog']['topN'])
        if default_model_cfg['DeepLog']['use_pretrained_model']:
            model_path = default_model_cfg['DeepLog']['pretrained__deeplog_model_path']
        else:
            model_path = model.fit(xy_train)
    else:
        print("No sequence model: {}".format(model_name))
        exit()

    # 3 test
    y = OrderedDict()
    for name, seqs in xy_test.items():
        logger.info(f'Testing data : {name}')
        y_predicted = model.predict(seqs, model_path)
        y[name] = y_predicted

    return y, time_dict

def frequency_model(model_name):
    # 1 load data
    x_train, vocab2idx, tf_idf, mean = dataloader.frequency_train(path_train, interval,
                                                                  default_model_cfg[model_name])
    x_test, time_dict = dataloader.frequency_test(path_test, interval, vocab2idx,
                                                  default_model_cfg[model_name], tf_idf, mean)

    # 2 train
    if model_name == 'PCA':
        model = PCA(n_components=default_model_cfg['PCA']['n_components'],
                    c_alpha=default_model_cfg['PCA']['c_alpha'])
    elif model_name == 'InvariantsMiner':
        model = InvariantsMiner(percentage=default_model_cfg['InvariantsMiner']['percentage'],
                                epsilon=default_model_cfg['InvariantsMiner']['epsilon'],
                                scale_list=default_model_cfg['InvariantsMiner']['scale_list'])
    elif model_name == 'LogClustering':
        model = LogClustering(max_dist=default_model_cfg['LogClustering']['max_dist'],
                              anomaly_threshold=default_model_cfg['LogClustering']['anomaly_threshold'],
                              mode=default_model_cfg['LogClustering']['mode'],
                              num_bootstrap_samples=default_model_cfg['LogClustering']['num_bootstrap_samples'])
    else:
        print("No frequency model: {}".format(model_name))
        exit()
    model.fit(x_train)

    # 3 test
    y = OrderedDict()
    for name, seqs in x_test.items():
        logger.info(f'Testing data : {name}')
        y_predicted = model.predict(seqs)
        y[name] = y_predicted

    return y, time_dict

def save_result(y, time_dict, result_dir):
    dataloader.save_result_to_csv(y,
                                  time_dict,
                                  result_dir,
                                  interval,
                                  timezone)
    mox.file.copy(os.path.join(result_dir, "submit.csv"),
                  os.path.join(Context.get_output_path(), "submit.csv"))


if __name__ == '__main__':
    model_name = 'LogClustering'

    parse_log()

    print("Using model: {}".format(model_name))
    if model_name in sequence_model_name:
        y, time_dict = sequence_model(model_name)
    elif model_name in frequency_model_name:
        y, time_dict = frequency_model(model_name)
    else:
        print("No model: {}".format(model_name))
        exit()

    save_result(y, time_dict, result_dir + model_name + '/')