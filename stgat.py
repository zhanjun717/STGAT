#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import torch
import torch.nn as nn

from modules import (
    InputLayer,
    StgatBlock,
    BiLSTMLayer,
    Forecasting_Model,
    ReconstructionModel,
)

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

class STGAT(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        embed_dim=None,
        layer_numb=2,
        lstm_n_layers=1,
        lstm_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(STGAT, self).__init__()

        layers1 = []
        layers2 = []
        layers3 = []

        self.layer_numb = layer_numb
        self.h_temp = []

        self.input_1 = InputLayer(n_features, 1)
        self.input_2 = InputLayer(n_features, 5)
        self.input_3 = InputLayer(n_features, 7)

        for i in range(layer_numb):
            layers1 += [StgatBlock(n_features, window_size, dropout, alpha, embed_dim)]
        for i in range(layer_numb):
            layers2 += [StgatBlock(n_features, window_size, dropout, alpha, embed_dim)]
        for i in range(layer_numb):
            layers3 += [StgatBlock(n_features, window_size, dropout, alpha, embed_dim)]

        self.stgat_1 = nn.Sequential(*layers1)
        self.stgat_2 = nn.Sequential(*layers2)
        self.stgat_3 = nn.Sequential(*layers3)

        self.bilstm = BiLSTMLayer(n_features * 3, lstm_hid_dim, lstm_n_layers, dropout)
        self.recon_model = ReconstructionModel(window_size, 2 * lstm_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x, fc_edge_index, tc_edge_index):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        fc_edge_index_sets = get_batch_edge_index(fc_edge_index[-1,:,:], x.shape[0], x.shape[2])
        tc_edge_index_sets = get_batch_edge_index(tc_edge_index[-1,:,:], x.shape[0], x.shape[1])

        x_1 = x
        x_2 = self.input_2(x)
        x_3 = self.input_3(x)

        for layer in range(self.layer_numb):
            if layer==0:
                h_cat_1 = x_1 + self.stgat_1[layer](x_1, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_2 = x_2 + self.stgat_2[layer](x_2, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_3 = x_3 + self.stgat_3[layer](x_3, fc_edge_index_sets, tc_edge_index_sets)
            else:
                h_cat_1 = h_cat_1 + self.stgat_1[layer](h_cat_1, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_2 = h_cat_2 + self.stgat_2[layer](h_cat_2, fc_edge_index_sets, tc_edge_index_sets)
                h_cat_3 = h_cat_3 + self.stgat_3[layer](h_cat_3, fc_edge_index_sets, tc_edge_index_sets)

        h_cat = torch.cat([h_cat_1, h_cat_2, h_cat_3], dim=2)

        out_end = self.bilstm(h_cat)
        h_end = out_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        recons = self.recon_model(h_end)

        return recons
