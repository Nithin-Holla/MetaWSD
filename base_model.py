from torch import nn

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
        self.linear1 = nn.Linear(
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

    def forward(self, input_tensor):
        embeds = self.embedding(input_tensor).permute(0, 2, 1)
        convolution_outputs = []
        for i in self.kernel_sizes:
            convolution = self.convolve['Convolve' + str(i)](embeds)
            convolution = self.relu(convolution)
            max_pooled = torch.max(convolution, dim=-1)[0]
            convolution_outputs.append(max_pooled)
        hidden = torch.cat(convolution_outputs, 1)
        dropout_1 = self.dropout(hidden)
        output = self.softmax(self.linear1(dropout_1))
        return hidden, output


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

        self.gru1 = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.gru2 = nn.GRU(
            input_size=self.hidden,
            hidden_size=self.hidden,
            batch_first=True
        )
        self.linear = nn.Linear(self.hidden, self.num_classes)

        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.softmax = nn.Softmax(-1)
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
        gru_1, _ = self.gru1(embeds)
        gru_1 = self.tanh(gru_1)
        dropout_1 = self.dropout(gru_1)
        _, h_n = self.gru2(dropout_1)
        h_n = self.tanh(h_n)
        dropout_2 = self.dropout(h_n[0])
        output = self.linear(dropout_2)
        if output.size()[-1] > 1:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return h_n[0], output


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

        self.gru1 = nn.GRU(
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
        hidden, _ = self.gru1(embeds)
        hidden = self.tanh(hidden)
        d = self.tanh(self.linear1(hidden))
        dropout = self.dropout(d)
        output = self.linear2(dropout)
        if output.size()[-1] > 1:
            output = self.softmax(output)
        else:
            output = self.sigmoid(output)
        return output
