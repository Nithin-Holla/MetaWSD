from base_models import RNNClassificationModel
from torch import nn
from torch import optim

import coloredlogs
import copy
import logging
import os
import torch

logger = logging.getLogger('AbuseBaselineLog')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class AbuseBaseModel(nn.Module):
    def __init__(self, config):
        super(AbuseBaseModel, self).__init__()
        self.base = config['base']
        self.early_stopping = config['early_stopping']
        self.learner_loss = nn.CrossEntropyLoss()
        self.learner_lr = config.get('meta_lr', 1e-3)
        self.learner_lr_decay = config.get('meta_lr_decay', 0.0)
        self.learner = RNNClassificationModel(
            config['learner_params'], config['embeddings'], True
        )
        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_baseline'])
            ))

    def forward(self, train_loader, test_loader, dataset, updates=1):
        optimizer = optim.Adam(
            self.learner.parameters(), lr=self.learner_lr,
            weight_decay=self.learner_lr_decay
        )
        best_loss = float('inf')
        best_model = None
        patience = 0
        for epoch in range(updates):
            self.learner.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.learner(batch_x)
                loss = self.learner_loss(output, batch_y)
                loss.backward()
                optimizer.step()
            self.learner.eval()
            num_correct, num_total, total_loss = 0, 0, 0.0
            for batch_x, batch_y in test_loader:
                output = self.learner(batch_x)
                loss = self.learner_loss(output, batch_y)
                total_loss += loss.item()
                num_correct += torch.eq(
                    output.max(-1)[1], batch_y
                ).sum().item()
                num_total += batch_y.size()[0]
            logger.info('Language {}: loss = {:.5f} accuracy = {:.5f}'.format(
                dataset, total_loss, 1.0 * num_correct / num_total
            ))
            if total_loss < best_loss:
                patience = 0
                best_loss = total_loss
                best_model = copy.deepcopy(self.learner)
            else:
                patience += 1
                if patience == self.early_stopping:
                    break
        self.learner = copy.deepcopy(best_model)
