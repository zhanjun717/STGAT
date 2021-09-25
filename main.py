#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import argparse
import numpy as np
import random
import torch
import os
from stgat import STGAT
from build_data import dataloda, get_target_dims, MyJsonEncoder,str2bool
from torch import nn
import time
from pytool import EarlyStopping
from sklearn.metrics import precision_recall_curve, mean_squared_error, roc_curve, auc ,f1_score, precision_recall_fscore_support,confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from eval_methods import *
from tqdm import tqdm
import json
from datetime import datetime

class Main(object):
    def __init__(self,config):
        id = datetime.now().strftime("%d%m%Y_%H%M%S")

        self.config = config
        print(f'*************************** dataset:{config.dataset} ID:{id}*****************************')
        print(config)

        # 加载数据
        self.train_dataset, \
        self.test_dataset, \
        self.train_dataloader, \
        self.val_dataloader, \
        self.test_dataloader,\
        self.train_all_dataloader\
            = dataloda(config)

        # 构建输出及日志路径
        if config.dataset == 'SMD':
            output_path = f'output/SMD/{config.group}'
        elif config.dataset in ['MSL', 'SMAP', 'SWat', 'WADI', 'WT/WT03', 'WT/WT23']:
            output_path = f'output/{config.dataset}'
        else:
            raise Exception(f'Dataset "{config.dataset}" not available.')

        self.log_dir = f'{output_path}/logs'
        if self.config.batch_train_id != '':
            if self.config.batch_train_id_back == 'init':
                self.save_path = f"{output_path}/{config.batch_train_id}"
            else:
                self.save_path = f"{output_path}/{config.batch_train_id}/{id}"
        else:
            self.save_path = f"{output_path}/{id}"

        self.output_path = f'{output_path}'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # EarlyStopping
        self.early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.00001, store_path=self.save_path)

        # Save config
        args_path = f"{self.save_path}/config.txt"
        with open(args_path, "w") as f:
            json.dump(config.__dict__, f, indent=2)

        # 依据数据集获取网络输出的维度，对于MSL以及SMAP数据集，由于数据由连续的遥测数据以及one-hot的控制指令组成，
            # 所以在预测和重建过程中仅对遥测数据进行，输出维度为1
        self.n_features = self.train_dataset.raw_data.shape[1]   # 获取数据真实维度
        self.target_dims = get_target_dims(config.dataset)       # 根据数据集对输出维度进行调整
        if self.target_dims is None:
            out_dim = self.n_features
            print(f"Will forecast all {self.n_features} input features")
        elif type(self.target_dims) == int:
            print(f"Will forecast input feature: {self.target_dims}")
            out_dim = 1
        else:
            print(f"Will forecast input features: {self.target_dims}")
            out_dim = len(self.target_dims)

        # 实例化模型
        self.stgat = STGAT(
            self.n_features,
            config.slide_win,
            out_dim,
            kernel_size=config.kernel_size,
            layer_numb=config.layer_numb,
            lstm_n_layers=config.lstm_n_layers,
            lstm_hid_dim=config.lstm_hid_dim,
            recon_n_layers=config.recon_n_layers,
            recon_hid_dim=config.recon_hid_dim,
            dropout=config.dropout,
            alpha=config.alpha
        ).to(self.config.device)
        # print(self.stgat)

    def train(self):
        model = self.stgat

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        forecast_criterion = nn.MSELoss().to(self.config.device)
        recon_criterion = nn.MSELoss().to(self.config.device)

        train_loss_list = []

        for epoch in range(self.config.epoch):
            model.train()
            acu_loss = 0
            time_start = time.time()
            for x, y, attack_labels, fc_edge_index, tc_edge_index in self.train_dataloader:
                x, y, fc_edge_index, tc_edge_index = [item.float().to(self.config.device) for item in [x, y,fc_edge_index, tc_edge_index]]
                x = x.permute(0,2,1)
                y = y.unsqueeze(1)

                # 正向传播
                recons = model(x, fc_edge_index, tc_edge_index)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims] # 将输入数据处理成与重建输出相同的形状,如果为NASA数据集则只取第一个维度的数据
                    y = y[:, :, self.target_dims].squeeze(-1) # 将输入数据处理成与预测输出相同的形状,如果未NASA数据集则只取第一个维度的数据

                if y.ndim == 3:
                    y = y.squeeze(1)

                recon_loss = torch.sqrt(recon_criterion(x, recons)) # recon_criterion=nn.MSELoss()  + scipy.stats.entropy(x, recons)
                loss = recon_loss

                # 方向梯度下降
                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 根据误差函数求导
                optimizer.step()  # 进行一轮梯度下降计算

                acu_loss += loss.item()

            time_end = time.time()

            # validation
            model.eval()
            val_loss = []
            for x, y, attack_labels, fc_edge_index, tc_edge_index  in self.val_dataloader:
                x, y, fc_edge_index, tc_edge_index  = [item.float().to(self.config.device) for item in [x, y, fc_edge_index, tc_edge_index ]]
                x = x.permute(0,2,1)
                y = y.unsqueeze(1)

                # 正向传播
                recons = model(x, fc_edge_index, tc_edge_index)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if y.ndim == 3:
                    y = y.squeeze(1)

                recon_loss = torch.sqrt(recon_criterion(x, recons))

                loss = recon_loss
                val_loss.append(loss.detach().cpu().numpy())

            train_average_loss = acu_loss / len(self.train_dataloader)
            train_loss_list.append(np.atleast_2d(np.array(train_average_loss)))

            val_mean_loss = np.mean(np.array(val_loss))
            print('# epoch:{}/{} , train loss:{} , val loss :{} , cost:{}'.format
                  (epoch, self.config.epoch, train_average_loss, val_mean_loss, (time_end - time_start)))

            self.early_stopping(val_mean_loss, model)
            if self.early_stopping.early_stop:
                print('Early stopping')
                train_loss = np.concatenate(train_loss_list)
                np.save(f"{self.save_path}/train_loss.npy",train_loss)
                break

    def predict(self, dataloader):
        model = self.stgat
        eval_loss = 0
        model.eval()
        predict = []
        recons = []
        test_label = []
        test_data = []
        loss_function = nn.MSELoss().to(self.config.device)

        with torch.no_grad():
            for x, y, attack_labels, fc_edge_index, tc_edge_index in tqdm(dataloader):
                x, y, fc_edge_index, tc_edge_index = [item.float().to(self.config.device) for item in [x, y, fc_edge_index, tc_edge_index]]
                x = x.permute(0, 2, 1)
                y_ = y.unsqueeze(1)

                # Shifting input to include the observed value (y) when doing the reconstruction
                recon_x = torch.cat((x[:, 1:, :], y_), dim=1)
                window_recon = model(recon_x, fc_edge_index, tc_edge_index)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y_[:, :, self.target_dims].squeeze(-1)

                # Extract last reconstruction only
                recons.append(window_recon[:, -1, :].detach().cpu().numpy()) # 重建后的数据，只取最后时刻点的一条记录 torch.Size([190, 15, 1]) ——>torch.Size([190, 1])

                test_label.append(attack_labels.cpu().numpy())
                test_data.append(y.cpu().numpy())

        return recons, test_data, test_label

    # 根据实际值和预测值求取每条记录的分数
    def get_score(self, recons, actual):
        actual = np.concatenate(actual, axis=0)
        recons = np.concatenate(recons, axis=0)

        anomaly_scores = np.zeros_like(actual)
        anomaly_original_scores = np.zeros_like(actual)
        df = pd.DataFrame()
        for i in range(actual.shape[1]):
            df[f"Recon_{i}"] = recons[:, i]
            df[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((recons[:, i] - actual[:, i])**2)
            anomaly_original_scores[:, i] = a_score

            epsilon = 1
            if self.config.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (epsilon + iqr)

           # anomaly_scores[:, i] = self.score_smooth(a_score)
            anomaly_scores[:, i] = abs(a_score)
            df[f"A_Score_{i}"] = anomaly_scores[:, i]

        anomaly_scores_mean = np.mean(anomaly_scores, 1)
        anomaly_original_scores = np.mean(anomaly_original_scores, 1)
        df['A_Score_Global'] = anomaly_scores_mean
        df['A_Score_orig_Global'] = anomaly_original_scores

        return df, anomaly_scores

    def adjust_anomaly_scores(self, scores, dataset, is_train, lookback, scale=True):
        """
        Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
        :param scores: anomaly_scores
        :param dataset: name of dataset
        :param is_train: if scores is from train set
        :param lookback: lookback (window size) used in model
        """

        # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
        if dataset.upper() not in ['SMAP', 'MSL']:
            return scores

        adjusted_scores = scores.copy()
        if is_train:
            md = pd.read_csv(f'./data/{dataset}/{dataset.lower()}_train_md.csv')
        else:
            md = pd.read_csv(f'./data/{dataset}/labeled_anomalies.csv')
            md = md[md['spacecraft'] == dataset.upper()]

        md = md[md['chan_id'] != 'P-2']

        # Sort values by channel
        md = md.sort_values(by=['chan_id'])

        # Getting the cumulative start index for each channel
        sep_cuma = np.cumsum(md['num_values'].values) - lookback
        sep_cuma = sep_cuma[:-1]
        buffer = np.arange(1, 20)
        i_remov = np.sort(np.concatenate((sep_cuma, np.array([i + buffer for i in sep_cuma]).flatten(),
                                          np.array([i - buffer for i in sep_cuma]).flatten())))
        i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
        i_remov = np.sort(np.unique(i_remov))
        if len(i_remov) != 0:
            adjusted_scores[i_remov] = 0

        # Normalize each concatenated part individually
        sep_cuma = np.cumsum(md['num_values'].values) - lookback
        s = [0] + sep_cuma.tolist()
        for c_start, c_end in [(s[i], s[i + 1]) for i in range(len(s) - 1)]:
            e_s = adjusted_scores[c_start: c_end + 1]

            if scale and e_s.size != 0:
                e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
            adjusted_scores[c_start: c_end + 1] = e_s

        return adjusted_scores

    def score_smooth(self, err_scores):
        smoothed_err_scores = np.zeros(err_scores.shape)
        before_num = 3
        for i in range(before_num, len(err_scores)):
            smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

        return smoothed_err_scores

    def model_metrics(self, test_recons, X_test, test_label, val_recons, X_val, with_adjust=True):
        print('Calculate Score...')
        test_y = np.concatenate(test_label)

        val_pred_df, val_anomaly_scores = self.get_score(val_recons, X_val)
        test_pred_df, test_anomaly_scores = self.get_score(test_recons, X_test)

        np.save(f"{self.save_path}/val_anomaly_scores.npy", val_anomaly_scores)
        np.save(f"{self.save_path}/test_anomaly_scores.npy", test_anomaly_scores)

        # 如果分数进行了归一化
        test_score = test_pred_df['A_Score_Global'].values
        val_score = val_pred_df['A_Score_Global'].values
        #
        # test_score = self.score_smooth(test_score)
        # val_score = self.score_smooth(val_score)

        # 在gat-pytorch里面有个标签调整的操作，即在数据拼接的位置对分数进行调整。
        val_score = self.adjust_anomaly_scores(val_score, self.config.dataset, True, self.config.slide_win,True)
        test_score = self.adjust_anomaly_scores(test_score, self.config.dataset, False, self.config.slide_win,True)

        test_y = test_y[:test_score.shape[0]].astype(int)

        # Best f1
        if test_y is not None:
            bf_eval = bf_search(test_score, test_y, start=0.00001, end=2, step_num=2000, verbose=False, adjust=with_adjust)
        else:
            bf_eval = {}

        m_eval = get_val_performance_data(test_anomaly_scores, val_anomaly_scores, test_label, adjust=with_adjust, topk=1)

        return bf_eval, m_eval

    def eval_to_float(self, bf_eval, m_eval):
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)
        for k, v in m_eval.items():
            if not type(m_eval[k]) == list:
                m_eval[k] = float(v)

        return bf_eval, m_eval

    def run(self):
        if self.config.batch_train_id_back == self.config.batch_train_id:
            print('Model exist!')
            self.stgat = torch.load(self.output_path + '/' + self.config.batch_train_id_back + '/checkpoint_best.pth').to(self.config.device)
        else:
            print('Train the model!')
            self.train()

        # 依据训练好的模型依此对测试集及验证集进行预测，得到预测值
        print("Predicting and calculating anomaly scores..")
        test_recons, X_test, test_label = self.predict(self.test_dataloader)
        train_recons, X_train, train_label = self.predict(self.train_all_dataloader)

        # 保存输出
        np.save(f"{self.save_path}/test_actual.npy", np.concatenate(X_test))
        np.save(f"{self.save_path}/test_recons.npy", np.concatenate(test_recons))
        np.save(f"{self.save_path}/test_label.npy", np.concatenate(test_label))

        np.save(f"{self.save_path}/train_actual.npy", np.concatenate(X_train))
        np.save(f"{self.save_path}/train_recons.npy", np.concatenate(train_recons))
        np.save(f"{self.save_path}/train_label.npy", np.concatenate(train_label))

        # 依据预测值、实际值以及实际值的标签求取评分
        bf_eval, m_eval = self.model_metrics(test_recons, X_test, test_label,
                                                             train_recons, X_train, self.config.with_adjust)

        print(f'=========================** Result for dataset:{self.config.dataset} **============================')
        print(f"Model config information:\n {self.config}")
        print(f"Results using best f1 score search:\n {bf_eval}")
        print(f"Results using max val-score method:\n {m_eval}")

        # Save
        bf_eval, m_eval = self.eval_to_float(bf_eval, m_eval)
        summary = {"bf_result": bf_eval, "max_val/train_result": m_eval}
        with open(f"{self.save_path}/summary_file.txt", "w") as f:
            json.dump(summary, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=1)
    parser.add_argument('-lr', help='learing rate', type=int, default=1e-3)

    parser.add_argument('-slide_win', help='slide_win', type=int, default=30)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='WT/WT23')
    parser.add_argument("-group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument('-normalize', type=str2bool, default=True)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-smooth', help='Kalman', type=str, default='Kalman')

    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=12)

    parser.add_argument('-load_model_path', help='output id', type=str, default='02092021_233355')
    parser.add_argument('-scale_scores', type=str2bool, default=True) # 是否对score进行标准化

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("-kernel_size", type=int, default=7)
    # STGAT layers
    parser.add_argument("-layer_numb", type=int, default=2)

    # LSTM layer
    parser.add_argument("-lstm_n_layers", type=int, default=2)
    parser.add_argument("-lstm_hid_dim", type=int, default=64)
    # Forecasting Model
    parser.add_argument("-fc_n_layers", type=int, default=3)
    parser.add_argument("-fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    parser.add_argument("-recon_n_layers", type=int, default=2)
    parser.add_argument("-recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("-alpha", type=float, default=0.2)
    parser.add_argument("-dropout", type=float, default=0.3)

    # 评估算法参数
    parser.add_argument("-dynamic_pot", type=str2bool, default=False)
    parser.add_argument("-level", type=float, default=None)
    parser.add_argument("-q", type=float, default=None)
    parser.add_argument("-gamma", type=float, default=1)
    parser.add_argument("-with_adjust", type=str2bool, default=False)

    parser.add_argument('-batch_train_id', help='train id', type=str, default='25092021_115849')
    parser.add_argument('-batch_train_id_back', help='train id', type=str, default='init')

    args = parser.parse_args()

    random.seed(args.random_seed)               # 设置随机数生成器的种子，是每次随机数相同
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)         # 为CPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed(args.random_seed)    # 为GPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False      # 网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速，适用场景是网络结构以及数据结构固定
    torch.backends.cudnn.deterministic = True   # 为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    os.environ['PYTHONHASHSEED'] = str(args.random_seed) # 为0则禁止hash随机化，使得实验可复现。

    main = Main(args)
    main.run()
