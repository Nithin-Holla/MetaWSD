from old.base_models import RNNEncoder
from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class AbuseProtoModel(nn.Module):
    def __init__(self, config):
        super(AbuseProtoModel, self).__init__()
        self.base = config['base']
        self.early_stopping = config['early_stopping']
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.encoder = RNNEncoder(
            config['learner_params'], config['embeddings'], embeds_grad=True
        )
        if config['trained_learner']:
            self.encoder.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_learner'])
            ))
        self.optimizer = optim.Adam(
            self.encoder.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )

    def forward(self, support_loaders, query_loaders, datasets, updates=1):
        query_losses = []
        query_accuracies = []
        for support_loader, query_loader, dataset in zip(
                support_loaders, query_loaders, datasets
        ):
            query_loss = 0.0
            query_accuracy = 0.0
            for epoch in range(updates):
                self.encoder.train()
                self.optimizer.zero_grad()
                x, y, l = [], [], []
                for batch_x, batch_y, batch_l in support_loader:
                    x.append(batch_x)
                    y.append(batch_y)
                    l.append(batch_l)
                x = flatten(torch.stack(tuple(x)))
                y = flatten(torch.stack(tuple(y)))
                l = flatten(torch.stack(tuple(l)))
                support = self.encoder(x, l)
                prototypes = build_prototypes(support, y)
                # Run on query
                self.encoder.eval()
                x, y, l = [], [], []
                for batch_x, batch_y, batch_l in query_loader:
                    x.append(batch_x)
                    y.append(batch_y)
                    l.append(batch_l)
                x = flatten(torch.stack(tuple(x)))
                y = flatten(torch.stack(tuple(y)))
                l = flatten(torch.stack(tuple(l)))
                query = self.encoder(x, l)
                output = normalized_distances(prototypes, query)
                query_loss = self.loss_fn(output, y)
                query_loss.backward()
                self.optimizer.step()
                # Calculate metrics
                num_correct = torch.eq(output.max(-1)[1], y).sum().item()
                num_total = y.size()[0]
                query_accuracy = num_correct / num_total
                logger.info(
                    (
                        'Dataset {}:\tloss={:.5f}\taccuracy={:.5f}'
                    ).format(dataset, query_loss, query_accuracy)
                )
            query_losses.append(query_loss)
            query_accuracies.append(query_accuracy)
        return query_losses, query_accuracies


def flatten(t):
    s = (t.size(0) * t.size(1),) + t.size()[2:]
    return t.reshape(s)


def build_prototypes(s, y):
    idx = {}
    for i, v in enumerate(y):
        if v.item() in idx:
            idx[v.item()].append(i)
        else:
            idx[v.item()] = [i]
    idx = [torch.LongTensor(idx[k]) for k in sorted(idx.keys())]
    return [s[i].mean(dim=0) for i in idx]


def normalized_distances(prototypes, q):
    d = torch.stack(
        tuple([q.sub(p).pow(2).sum(dim=-1) for p in prototypes]),
        dim=-1
    )
    return d.neg()
