from torch import nn


class RNNSequenceModel(nn.Module):
    def __init__(self, model_params, embeds=None, embeds_grad=False):
        super(RNNSequenceModel, self).__init__()
        self.hidden = model_params['hidden_size']
        self.embed_dim = model_params['embed_dim']
        self.dropout_ratio = model_params.get('dropout_ratio', 0.5)

        self.gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden,
            batch_first=True,
        )
        self.linear1 = nn.Linear(self.hidden, self.hidden // 2)

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
        self.gru.flatten_parameters()
        hidden, _ = self.gru(input_tensor)
        hidden = self.tanh(hidden)
        d = self.tanh(self.linear1(hidden))
        dropout = self.dropout(d)
        return dropout