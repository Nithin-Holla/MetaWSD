import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, AlbertModel


class RNNSequenceModel(nn.Module):
    def __init__(self, model_params):
        super(RNNSequenceModel, self).__init__()
        self.hidden_size = model_params['hidden_size']
        self.embed_dim = model_params['embed_dim']
        self.dropout_ratio = model_params.get('dropout_ratio', 0)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size // 4)

        self.dropout = nn.Dropout(p=self.dropout_ratio)
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

    def forward(self, input, input_len):
        self.gru.flatten_parameters()
        packed = pack_padded_sequence(input, input_len, batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)
        hidden = self.tanh(hidden)
        d = self.tanh(self.linear(hidden))
        dropout = self.dropout(d)
        return dropout


class MLPModel(nn.Module):

    def __init__(self, model_params):
        super(MLPModel, self).__init__()
        self.embed_dim = model_params['embed_dim']
        self.hidden_size = model_params['hidden_size']
        self.dropout_ratio = model_params.get('dropout_ratio', 0)
        self.linear = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_ratio))

    def forward(self, input, *args):
        out = self.linear(input)
        return out


class BERTSequenceModel(nn.Module):

    def __init__(self, model_params):
        super(BERTSequenceModel, self).__init__()
        self.embed_dim = model_params['embed_dim']
        self.hidden_size = model_params['hidden_size']
        self.dropout_ratio = model_params.get('dropout_ratio', 0)
        self.n_tunable_layers = model_params.get('fine_tune_layers', None)
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.linear = nn.Sequential(nn.Linear(self.embed_dim, self.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(p=self.dropout_ratio))

        self.bert.pooler.dense.weight.requires_grad = False
        self.bert.pooler.dense.bias.requires_grad = False

        if self.n_tunable_layers is not None:
            tunable_layers = {str(l) for l in range(12 - self.n_tunable_layers, 12)}
            for name, param in self.bert.named_parameters():
                if not set.intersection(set(name.split('.')), tunable_layers):
                    param.requires_grad = False

    def forward(self, input, input_len):
        attention_mask = (input.detach() != 0).float()
        output, _ = self.bert(input, attention_mask=attention_mask)
        output = output[:, 1:-1, :]  # Ignore the output of the CLS and SEP tokens
        output = self.linear(output)
        return output


class ALBERTRelationModel(nn.Module):

    def __init__(self, model_params):
        nn.Module.__init__(self)
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.cat_entity_rep = model_params.get('cat_entity_rep', True)
        self.embed_dim = model_params.get('embed_dim', 768)
        if self.cat_entity_rep:
            self.embed_dim *= 2
            self.albert.pooler.weight.requires_grad = False
            self.albert.pooler.bias.requires_grad = False

    def forward(self, inputs):
        if not self.cat_entity_rep:
            _, x = self.albert(inputs['word'], attention_mask=inputs['mask'])
            return x
        else:
            outputs = self.albert(inputs['word'], attention_mask=inputs['mask'])
            tensor_range = torch.arange(inputs['word'].size()[0])
            h_state = outputs[0][tensor_range, inputs['pos1']]
            t_state = outputs[0][tensor_range, inputs['pos2']]
            state = torch.cat((h_state, t_state), -1)
            return state
