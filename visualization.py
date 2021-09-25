#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, mean_squared_error, f1_score, precision_recall_fscore_support,confusion_matrix, precision_score, recall_score, roc_auc_score


def get_topk_scores(total_err_scores, topk=1):
    total_err_scores = total_err_scores.T
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]

    # 特征最大分数值
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    return total_topk_err_scores

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_err_scores = total_err_scores.T
    normal_scores = normal_scores.T
    gt_labels = gt_labels.tolist()
    total_features = total_err_scores.shape[0]
    # 取出最大的topK个scores值的索引值,得到每条样本中分数最大的特征索引值
    topk_indices = np.argpartition(total_err_scores, range(total_features - topk - 1, total_features), axis=0)[
                   -topk:]
    total_topk_err_scores = []
    topk_err_score_map = []
    # 取出最大的topK个scores值
    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    # 得到阈值
    thresold = np.max(normal_scores)

    # 根据标签得到最终预测的label
    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels)
    C = confusion_matrix(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return {
        "f1": f1,
        "precision": pre,
        "recall": rec,
        "TP": C[0, 0],
        "TN": C[1, 1],
        "FP": C[0, 1],
        "FN": C[1, 0],
        "threshold": thresold,
        "latency": 0,
        "roc_auc": auc_score,
        "pred_labels": pred_labels,
    }

if __name__ == '__main__':
    path = "11082021_112415"
    test_predicted = np.load(f'./output/SWat/{path}/test_predict.npy')
    test_true = np.load(f'./output/SWat/{path}/test_actual.npy')
    test_recons = np.load(f'./output/SWat/{path}/test_recons.npy')
    test_labels = np.load(f'./output/SWat/{path}/test_label.npy')

    train_predicted = np.load(f'./output/SWat/{path}/train_predict.npy')
    train_recons = np.load(f'./output/SWat/{path}/train_recons.npy')
    train_true = np.load(f'./output/SWat/{path}/train_actual.npy')
    train_labels = np.load(f'./output/SWat/{path}/train_label.npy')

    val_anomaly_scores = np.load(f'./output/SWat/{path}/val_anomaly_scores.npy')
    test_anomaly_scores = np.load(f'./output/SWat/{path}/test_anomaly_scores.npy')

    val_total_topk_err_scores = get_topk_scores(val_anomaly_scores)
    test_total_topk_err_scores = get_topk_scores(test_anomaly_scores)

    m_eval = get_val_performance_data(test_anomaly_scores, val_anomaly_scores, test_labels, topk=1)
    pred_labels = m_eval["pred_labels"]

    # test_scores = np.load('./output/SWat/03082021_194902/test_anomaly_scores.npy')
    # total_topk_err_scores = np.load('./output/SWat/03082021_194902/test_score.npy')

    # for i in range(len(test_true.columns)):
    #     plt.figure()
    #     plt.scatter(np.arange(len(train_true)), train_true[:,i], c='red', label='train_true', s=2)
    #     plt.scatter(np.arange(len(train_predicted)), train_predicted[:,i], c='blue', label='train_predicted', s=2)
    #     plt.legend(loc='upper left', fontsize=8)
    #     plt.show()

    # for i in range(train_true.shape[1]):
    #     plt.figure()
    #     plt.scatter(np.arange(len(train_true)), train_true[:,i], c='red', label='train_true', s=2)
    #     plt.scatter(np.arange(len(train_predicted)), train_predicted[:,i], c='blue', label='train_predicted', s=2)
    #     plt.scatter(np.arange(len(train_recons)), train_recons[:,i], c='green', label='train_recons', s=2)
    #     plt.legend(loc='upper left', fontsize=8)
    #     plt.show()

    for i in range(test_true.shape[1]):
        plt.figure()
        plt.scatter(np.arange(len(test_true)), test_true[:,i], c='red', label='test_true', s=2)
        plt.scatter(np.arange(len(test_predicted)), test_predicted[:,i], c=test_labels, label='test_predicted', s=2)
        plt.scatter(np.arange(len(test_recons)), test_recons[:,i], c=test_labels, label='test_recons', s=2)
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

        plt.figure()
        plt.scatter(np.arange(len(test_total_topk_err_scores)), np.atleast_2d(test_total_topk_err_scores).T, c=test_labels, s=2)
        plt.legend(['test_scores'])
        plt.show()

        plt.figure()
        plt.scatter(np.arange(len(test_total_topk_err_scores)), np.atleast_2d(test_total_topk_err_scores).T, c=pred_labels, s=2)
        plt.legend(['test_scores'])
        plt.show()

    # plt.figure()
    # plt.scatter(np.arange(len(total_topk_err_scores)), total_topk_err_scores, c=test_labels, s=2)
    # plt.legend(['total_topk_err_scores'])
    # plt.show()
    #
    # total_topk_err_scores_ = get_topk_scores(test_scores, 2)
    # plt.figure()
    # plt.scatter(np.arange(len(total_topk_err_scores_)), total_topk_err_scores_, c=test_labels, s=2)
    # plt.legend(['total_topk_err_scores_'])
    # plt.show()

    # plt.figure(0)
    # plt.subplot(411)
    # plt.scatter(np.arange(len(test_predicted)), test_predicted[:,0], c=test_labels[:,0], s=2)
    #
    # plt.subplot(412)
    # plt.scatter(np.arange(len(test_true)), test_true[:,0], c=test_labels[:,0], s=2)
    #
    # plt.subplot(413)
    # plt.scatter(np.arange(len(total_topk_err_scores)), total_topk_err_scores, c=test_labels[:,0], s=2)
    #
    # plt.subplot(414)
    # plt.scatter(np.arange(len(test_scores)), test_scores[:,0], c=test_labels[:,0], s=2)
    #
    # plt.show()
    #
    # train_orig = pd.read_csv(f'./data/SWat/train.csv', sep=',', header=1)
    # test_orig = pd.read_csv(f'./data/SWat/test.csv', sep=',')
    #
    # train_orig[' Timestamp'] = pd.to_datetime(train_orig[' Timestamp'])
    # th_time = train_orig[' Timestamp'][0] + pd.Timedelta('6H')  # 两个datetime值之间的差(如日,秒和微妙)
    # train_orig = train_orig[train_orig[' Timestamp'] > th_time]
    #
    # train, test = train_orig.iloc[:100000, :], test_orig.iloc[:50000, :]
    #
    # plt.figure(1)
    # plt.subplot(311)
    # plt.scatter(np.arange(len(test)), test[:,0], c=test[:,-1], s=2)
    #
    # plt.subplot(312)
    # plt.scatter(np.arange(len(test)), test[:,1], c=test_labels[:,-1], s=2)
    #
    # plt.subplot(313)
    # plt.scatter(np.arange(len(test)), test[:,2], c=test_labels[:,-1], s=2)
    #
    # plt.show()

