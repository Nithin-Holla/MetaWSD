from torch import nn
from torch.nn import functional as func
from torch.nn.utils.rnn import pack_padded_sequence

import numpy
import os
import random
import torch
os.environ['PYTHONHASHSEED'] = '0'
numpy.random.seed(57)
random.seed(75)


class CNNClassificationModel(nn.Module):
    def __init__(self, model_params, embeds=None, embeds_grad=False):
        super(CNNClassificationModel, self).__init__()
        self.kernel_sizes = model_params['kernel_sizes']
        self.num_filters = model_params['num_filters']
        self.embed_dim = model_params.get('embed_dim', 300)
        self.num_classes = model_params.get('num_classes', 2)
        self.dropout_ratio = model_params.get('dropout_ratio', 0.5)

        self.embedding = nn.Embedding(
            num_embeddings=model_params['vocab_size'],
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)
        self.embedding.weight.requires_grad = embeds_grad

        self.convolve = nn.ModuleDict()
        self.pool = nn.ModuleDict()
        for i in self.kernel_sizes:
            self.convolve['Convolve' + str(i)] = nn.Conv1d(
                in_channels=self.embed_dim,
                out_channels=self.num_filters,
                kernel_size=i
            )
        self.linear = nn.Linear(
            self.num_filters * len(self.kernel_sizes), self.num_classes
        )

        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'embedding' not in name:
                nn.init.xavier_uniform_(param)

    def normal_forward(self, input_tensor):
        embeds = self.embedding(input_tensor).permute(0, 2, 1)
        convolution_outputs = []
        for i in self.kernel_sizes:
            convolution = self.convolve['Convolve' + str(i)](embeds)
            convolution = self.relu(convolution)
            max_pooled = torch.max(convolution, dim=-1)[0]
            convolution_outputs.append(max_pooled)
        hidden = torch.cat(tuple(convolution_outputs), 1)
        dropout = self.dropout(hidden)
        output = self.softmax(self.linear(dropout))
        return output

    def forward(self, input_tensor, weights=None, train=True):
        if weights is None:
            return self.normal_forward(input_tensor)
        embeds = func.embedding(
            input_tensor, weights['embedding.weight'], padding_idx=0,
        ).permute(0, 2, 1)
        convolution_outputs = []
        for i in self.kernel_sizes:
            convolution = func.conv1d(
                embeds, weights['convolve.Convolve{}.weight'.format(i)],
                bias=weights['convolve.Convolve{}.bias'.format(i)]
            )
            convolution = func.relu(convolution)
            max_pooled = torch.max(convolution, dim=-1)[0]
            convolution_outputs.append(max_pooled)
        concat = torch.cat(tuple(convolution_outputs), 1)
        dropout = func.dropout(concat, p=self.dropout.p, training=train)
        hidden = func.linear(
            dropout, weights['linear.weight'], weights['linear.bias']
        )
        return func.softmax(hidden, dim=-1)


class RNNClassificationModel(nn.Module):
    def __init__(self, model_params, embeds=None, embeds_grad=False):
        super(RNNClassificationModel, self).__init__()
        self.embed_dim = model_params['embed_dim']
        self.hidden = model_params['hidden_size']
        self.num_classes = model_params['num_classes']
        self.dropout_ratio = model_params.get('dropout_ratio', 0.5)

        self.embedding = nn.Embedding(
            num_embeddings=model_params['vocab_size'],
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)
        self.embedding.weight.requires_grad = embeds_grad

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.linear = nn.Linear(self.hidden, self.num_classes)

        self.dropout = nn.Dropout(p=self.dropout_ratio)
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

    def forward(self, input_tensor, lengths):
        embeds = self.embedding(input_tensor)
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        _, h_n = self.gru(embeds)
        h_n = self.tanh(h_n)
        dropout_1 = self.dropout(h_n[0])
        output = self.linear(dropout_1)
        if output.size()[-1] > 1:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return output


class RNNSequenceModel(nn.Module):
    def __init__(self, model_params, embeds=None, embeds_grad=False):
        super(RNNSequenceModel, self).__init__()
        self.hidden = model_params['hidden_size']
        self.num_outputs = model_params['num_outputs']
        self.embed_dim = model_params['embed_dim']
        self.dropout_ratio = model_params.get('dropout_ratio', 0.5)

        self.embedding = nn.Embedding(
            num_embeddings=model_params['vocab_size'],
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)
        self.embedding.weight.requires_grad = embeds_grad

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.linear1 = nn.Linear(self.hidden, self.hidden // 2)
        self.linear2 = nn.Linear(self.hidden // 2, self.num_outputs)

        self.dropout = nn.Dropout(p=self.dropout_ratio)
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
        hidden, _ = self.gru(embeds)
        hidden = self.tanh(hidden)
        d = self.tanh(self.linear1(hidden))
        dropout = self.dropout(d)
        output = self.linear2(dropout)
        if output.size()[-1] > 1:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return output


class RNNEncoder(nn.Module):
    def __init__(self, model_params, embeds=None, embeds_grad=False):
        super(RNNEncoder, self).__init__()
        self.embed_dim = model_params['embed_dim']
        self.hidden = model_params['hidden_size']

        self.embedding = nn.Embedding(
            num_embeddings=model_params['vocab_size'],
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        if embeds is not None:
            self.embedding.weight = nn.Parameter(embeds)
        self.embedding.weight.requires_grad = embeds_grad

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        for name, param in self.named_parameters():
            if 'embedding' in name:
                continue
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'hh' not in name:
                nn.init.xavier_uniform_(param)
            elif 'weight' in name and 'hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, input_tensor, lengths):
        embeds = self.embedding(input_tensor)
        embeds = pack_padded_sequence(embeds, lengths, batch_first=True)
        _, h_n = self.gru(embeds)
        return h_n[0]
