import numpy
import os
import random
import torch
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(57)
random.seed(75)
torch.manual_seed(1025)

from torch import nn
from torch.optim import Adam
from torch.utils import data

import logging
import pickle


class CNNModel(nn.Module):
    def __init__(self, v_size, emb_dim, num_class, embeds=None):
        super(CNNModel, self).__init__()
        self.kernel_sizes = (2, 3, 4)
        self.num_filters = 300

        self.embedding = nn.Embedding(
            num_embeddings=v_size,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)

        self.convolve = nn.ModuleDict()
        self.pool = nn.ModuleDict()
        for i in self.kernel_sizes:
            self.convolve['Convolve' + str(i)] = nn.Conv1d(
                in_channels=emb_dim,
                out_channels=self.num_filters,
                kernel_size=i
            )
        self.linear1 = nn.Linear(self.num_filters * len(self.kernel_sizes), num_class)

        self.dropout_50 = nn.Dropout(p=0.50)
        self.softmax = nn.Softmax(1)
        self.relu = nn.ReLU()

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)

    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor).permute(0, 2, 1)
        convolution_outputs = []
        for i in self.kernel_sizes:
            convolution = self.convolve['Convolve' + str(i)](embeds)
            convolution = self.relu(convolution)
            max_pooled = torch.max(convolution, dim=-1)[0]
            convolution_outputs.append(max_pooled)
        hidden = torch.cat(convolution_outputs, 1)
        dropout_1 = self.dropout_50(hidden)
        output = self.softmax(self.linear1(dropout_1))
        return hidden, output


class RNNClassificationModel(nn.Module):
    def __init__(self, v_size, emb_dim, hid_dim, num_class, embeds=None):
        super(RNNClassificationModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=v_size,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)

        self.gru1 = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=hid_dim,
            hidden_size=hid_dim,
            batch_first=True
        )
        self.linear = nn.Linear(hid_dim, num_class)

        self.dropout_25 = nn.Dropout(p=0.25)
        self.dropout_50 = nn.Dropout(p=0.50)
        self.softmax = nn.Softmax(1)
        self.tanh = nn.Tanh()

        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'hh' not in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and 'hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor)
        dropout_1 = self.dropout_25(embeds)
        gru_1, _ = self.gru1(dropout_1)
        gru_1 = self.tanh(gru_1)
        dropout_2 = self.dropout_25(gru_1)
        _, h_n = self.gru2(dropout_2)
        h_n = self.tanh(h_n)
        dropout_3 = self.dropout_50(h_n[0])
        output = self.linear(dropout_3)
        if output.size()[-1] > 1:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return h_n[0], output


class RNNSequenceModel(nn.Module):
    def __init__(self, v_size, emb_dim, hid_dim, num_class, embeds=None):
        super(RNNSequenceModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=v_size,
            embedding_dim=emb_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)

        self.gru1 = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hid_dim, hid_dim // 2)
        self.linear2 = nn.Linear(hid_dim // 2, num_class)

        self.dropout_25 = nn.Dropout(p=0.25)
        self.dropout_50 = nn.Dropout(p=0.50)
        self.softmax = nn.Softmax(-1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'hh' not in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and 'hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor)
        dropout_1 = self.dropout_25(embeds)
        h, _ = self.gru1(dropout_1)
        h = self.tanh(h)
        d = self.tanh(self.linear1(h))
        o = self.linear2(d)
        if o.size()[-1] > 1:
            o = self.softmax(o)
        else:
            o = self.sigmoid(o)
        return o


class DataLoader(data.Dataset):
    def __init__(self, samples, classes):
        super(DataLoader, self).__init__()
        self.samples = samples
        self.classes = classes

    def __getitem__(self, index):
        return self.samples[index], self.classes[index]

    def __len__(self):
        return len(self.classes)
