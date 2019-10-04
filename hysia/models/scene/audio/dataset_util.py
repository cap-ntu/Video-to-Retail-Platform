import yaml
import csv
from glob import glob
import os.path as osp

this_dir = osp.dirname(__file__)
config_yaml = osp.join(this_dir, 'configs/dataset_config.yaml')
dataset_config = None


def load_yaml(yaml_path=config_yaml):
    with open(yaml_path, 'r') as yaml_file:
        return yaml.load(yaml_file)


def get_dataconfig(dataset_name):
    try:
        global dataset_config
        if dataset_config is None:
            dataset_config = load_yaml()['dataset']
        return dataset_config[dataset_name]
    except KeyError:
        print('Wrong Dataset Parameter')
        raise KeyError


def get_datalist(dataset_name):
    try:
        global dataset_config
        if dataset_config is None:
            dataset_config = load_yaml()
        dataset = dataset_config[dataset_name]
        label_dict = {}
        if dataset_name == 'dcase2016':
            label_dict = get_dcase2016_dict(dataset)
        elif dataset_name == 'dcase2018':
            label_dict = get_dcase2018_dict(dataset)
        file_list = list(label_dict.keys())
        label_list = list(label_dict.values())
        return dataset, label_dict
    except KeyError:
        print('Wrong Dataset Parameter')


def get_dcase2016_dict(dataset_dict):
    with open(dataset_dict['label_path'], 'rt') as f:
        reader = csv.reader(f)
        lis = list(reader)
        ret = {}
        labels = dataset_dict['labels']
        for li in lis:
            # load data
            [na, lb] = li[0].split('\t')
            na = na.split('/')[1][0:-4]
            ret[na] = labels.index(lb)
        return ret


def get_dcase2018_dict(dataset_dict):
    with open(dataset_dict['label_path'], 'rt') as f:
        reader = csv.reader(f)
        next(reader, None)
        lis = list(reader)
        ret = {}
        labels = dataset_dict['labels']
        for li in lis:
            # load data
            [na, lb, _, _] = li[0].split('\t')
            na = na.split('/')[1][0:-4]
            ret[na] = labels.index(lb)
        return ret
