#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import torch.nn.functional as F

class InputLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """
    def __init__(self, n_features, kernel_size=7):
        super(InputLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class StgatBlock(nn.Module):
    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None):
        super(StgatBlock, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.alpha = alpha
        self.embed_dim = embed_dim if embed_dim is not None else n_features

        self.embed_dim *= 2

        self.feature_gat_layers = GATConv(window_size, window_size)
        self.temporal_gat_layers = GATConv(n_features, n_features)

        self.temporal_gcn_layers = GCNConv(n_features, n_features)

    def forward(self, data, fc_edge_index, tc_edge_index):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps
        # ft = data.clone().detach()
        # tp = data.clone().detach()

        # ft = ft.permute(0, 2, 1)
        # batch_num, node_num, all_feature = ft.shape
        # ft = ft.reshape(-1, all_feature).contiguous()
        # f_out = self.feature_gat_layers(ft, fc_edge_index)
        # f_out = F.relu(f_out)
        # f_out = f_out.view(batch_num, node_num, -1)
        # f_out = f_out.permute(0, 2, 1)
        #
        # batch_num, node_num, all_feature = tp.shape
        # tp = tp.reshape(-1, all_feature).contiguous()
        # t_out = self.temporal_gat_layers(tp, tc_edge_index)
        # t_out = F.relu(t_out)
        # t_out = t_out.view(batch_num, node_num, -1)
        #
        # return f_out + t_out   #self.relu(res + h + z)

        x = data.clone().detach()
        x = x.permute(0, 2, 1)
        batch_num, node_num, all_feature = x.shape

        x = x.reshape(-1, all_feature).contiguous()
        # f_out = self.feature_gat_layers(x, fc_edge_index, return_attention_weights = True)
        f_out = self.feature_gat_layers(x, fc_edge_index)
        f_out = F.relu(f_out)
        f_out = f_out.view(batch_num, node_num, -1)
        f_out = f_out.permute(0, 2, 1)
        z = f_out.reshape(-1, node_num).contiguous()

        t_out = self.temporal_gcn_layers(z, tc_edge_index)
        t_out = F.relu(t_out)
        t_out = t_out.view(batch_num, node_num, -1)

        return t_out.permute(0, 2, 1)   #self.relu(res + h + z)

class BiLSTMLayer(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        out, h = self.bilstm(x)
        out = out.permute(1,0,2)[-1, :, :] #, h[-1, :, :]  # Extracting from last layer
        return out

class BiLSTMDecoder(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(BiLSTMDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.bilstm = nn.LSTM(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout, bidirectional=True)

    def forward(self, x):
        decoder_out, _ = self.bilstm(x)
        return decoder_out

class ReconstructionModel(nn.Module):
    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = BiLSTMDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(2 * hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        decoder_out = self.decoder(h_end_rep)
        out = self.fc(decoder_out)
        return out

class Forecasting_Model(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)