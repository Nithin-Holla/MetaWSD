from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNSequenceModel(nn.Module):
    def __init__(self, model_params):
        super(RNNSequenceModel, self).__init__()
        self.hidden = model_params['hidden_size']
        self.embed_dim = model_params['embed_dim']
        self.dropout_ratio = model_params.get('dropout_ratio', 0)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.linear1 = nn.Linear(2 * self.hidden, self.hidden // 2)

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
        d = self.tanh(self.linear1(hidden))
        dropout = self.dropout(d)
        return dropout
