from base_model import RNNSequenceModel
from meta_model import MetaModel
from torch import nn
from torch import optim

import torch


class MetaLearning:
    def __init__(self, config):
        if 'rnn_sequence' in config['learner_model']:
            config['learner_model'] = RNNSequenceModel(config['learner_params'])
            config['learner_loss'] = nn.CrossEntropyLoss()

        self.meta_model = MetaModel(config)
        self.config = config

    def meta_training(self, support_loaders, query_loaders):
        meta_optimizer = optim.Adam(self.meta_model.learner.parameters())

        best_loss = float('inf')
        patience = 0
        for epoch in range(self.config['num_meta_epochs']):
            meta_optimizer.zero_grad()
            loss, _ = self.meta_model(support_loaders, query_loaders)
            meta_optimizer.step()

            loss_value = torch.sum(torch.Tensor(loss)).item()
            if loss_value <= best_loss:
                patience = 0
                best_loss = loss_value
            else:
                patience += 1
                if patience == self.config['early_stopping']:
                    break
        self.meta_model.learner.save('file')

    def meta_testing(self, support_loaders, query_loaders):
        pass
