#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd

from torch.utils.data import DataLoader, random_split, Subset

import random
from sklearn import preprocessing
import argparse
import json
from tsmoothie.smoother import *
from net_struct import get_feature_map, get_fc_graph_struc, get_tc_graph_struc
from preprocess import build_loc_net

def get_rolling_label(labels):
    count_normal = sum(labels == 0)
    count_attack = sum(labels == 1)
    if count_normal > count_attack:
        return 0.0
    else:
        return 1.0

def down_sample(train, test, T):
    train = train.rolling(T).median()[0::T]
    test_data = test.iloc[:, :-1].rolling(T).median()[0::T]
    test_label = test.iloc[:, -1:].rolling(T).apply(get_rolling_label)[0::T]

    return train.dropna(), pd.concat([test_data, test_label], axis=1).dropna()

class TimeDataset(Dataset):
    def __init__(self, raw_data,  edge_index, mode='train', config=None):
        self.raw_data = raw_data
        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        labels = []
        if 'train' in mode:
            x_data = raw_data
            labels.append([0]*x_data.shape[0])
        elif 'test' in mode:
            x_data = raw_data.iloc[:, :-1]  # 取出数据
            labels = raw_data.iloc[:,-1]  # 取出标签

        data = x_data

        # to tensor
        data = torch.tensor(np.array(data).T).double()
        labels = torch.tensor(np.array(labels).T).double()

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []
        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_train = self.mode == 'train'
        node_num, total_time_len = data.shape

        # 如果为训练数据集，则返回窗口起始位置到数据集末尾，步长为slide_stride的滑窗索引，如果为其他数据集则返回步长为1的滑窗索引
        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            ft = data[:, i - slide_win:i]  # 0~14条
            tar = data[:, i]  # 第15条

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        fc_edge_index = self.edge_index[0].long()
        tc_edge_index = self.edge_index[1].long()

        label = self.labels[idx].double()

        return feature, y, label, fc_edge_index, tc_edge_index

def get_loaders(train_dataset, seed, batch, val_ratio=0.1): # val_ratio 验证集比例
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len) # 返回指定递增基数集合中的一个随机数，随机取出一个数据作为验证集开始的位置，从此位置开始取验证集数据
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]]) # 得到除去验证集数据的训练集索引
    train_subset = Subset(train_dataset, train_sub_indices) # 获取指定一个索引序列对应的子数据集

    val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=True)

    return train_dataloader, val_dataloader

def get_adge_index(dataset, train, config):
    feature_map = get_feature_map(dataset)  # 获取特征
    fc_struc = get_fc_graph_struc(dataset)  # 获取所有节点与其他节点的连接关系字典

    fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)  # 获取所有节点与其子集节点的连接矩阵
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)  # 将连接矩阵转换成Tensor,torch.Size([2, 702])

    temporal_map = list(range(0, config.slide_win))
    tc_struc = get_tc_graph_struc(config.slide_win)

    tc_edge_index = build_loc_net(tc_struc, list(range(0, config.slide_win)),
                                  feature_map=temporal_map)  # 获取所有节点与其子集节点的连接矩阵
    tc_edge_index = torch.tensor(tc_edge_index, dtype=torch.long)  # 将连接矩阵转换成Tensor,torch.Size([2, 702])

    return (fc_edge_index, tc_edge_index)

def dataloda(config):
    dataset = config.dataset
    if 'WT' in dataset:
        train_orig = pd.read_csv(f'./data/{dataset}/train_orig.csv', sep=',').dropna(axis=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test_orig.csv', sep=',').dropna(axis=0)

        # train, test = train_orig.drop(columns='采样时间'), test_orig.drop(columns='采样时间')
        train, test = train_orig, test_orig

        test_label = test["attack"]
        dataset_label = np.array(test_label)

        test_columns = test.columns  # attack
        test.drop(["attack"], axis=1, inplace=True)

        train_columns = train.columns

        if config.normalize:
            train_normalizer = preprocessing.MinMaxScaler().fit(train)
            train = train_normalizer.transform(train)
            test = train_normalizer.transform(test)
        else:
            test = np.array(test)

        test = np.hstack((test, dataset_label[:, np.newaxis]))

        train = pd.DataFrame(train, columns=train_columns)
        test = pd.DataFrame(test, columns=test_columns)

        # train, test = down_sample(train, test, 10)

    edge_index = get_adge_index(dataset, train, config)

    cfg = {
        'slide_win': config.slide_win,
        'slide_stride': config.slide_stride,
    }

    train_dataset = TimeDataset(train, edge_index, mode='train', config=cfg)
    test_dataset = TimeDataset(test, edge_index, mode='test', config=cfg)

    train_dataloader, val_dataloader = get_loaders(train_dataset = train_dataset, seed = config.random_seed,
                                                   batch = config.batch, val_ratio=config.val_ratio)

    test_dataloader = DataLoader(test_dataset, batch_size=config.batch,
                                      shuffle=False, num_workers=0)

    train_all_dataloader = DataLoader(train_dataset, batch_size=config.batch,
                                      shuffle=False, num_workers=0)

    return train_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader, train_all_dataloader

def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP":
        return [0]
    elif dataset == "MSL":
        return [0]
    elif dataset == "SMD" or dataset == "SWat" or dataset == "WADI" or dataset == "WT/WT03" or dataset == "WT/WT23":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        """
        只要检查到了是bytes类型的数据就把它转为str类型
        :param obj:
        :return:
        """
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)
