from base_models import RNNClassificationModel
from torch import nn
from torch import optim

import coloredlogs
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
        self.stamp = config['stamp']
        self.early_stopping = config['early_stopping']
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.learner = RNNClassificationModel(
            config['learner_params'], config['embeddings'], True
        )
        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_baseline'])
            ))
        self.optimizer = optim.Adam(
            self.learner.parameters(), lr=self.lr,
            weight_decay=self.weight_decay
        )

    def forward(self, train_loader, test_loader, dataset, updates=1):
        model_path = os.path.join(
            self.base, 'models', 'Baseline-{}.h5'.format(self.stamp)
        )
        best_loss = float('inf')
        patience = 0
        for epoch in range(updates):
            self.learner.train()
            num_correct, num_total, train_loss = 0.0, 0.0, 0.0
            for batch_x, batch_y, batch_l in train_loader:
                self.optimizer.zero_grad()
                output = self.learner(batch_x, batch_l)
                loss = self.loss_fn(output, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                num_correct += torch.eq(
                    output.max(-1)[1], batch_y
                ).sum().item()
                num_total += batch_y.size()[0]
            train_accuracy = num_correct / num_total
            self.learner.eval()
            num_correct, num_total, test_loss = 0.0, 0.0, 0.0
            for batch_x, batch_y, batch_l in test_loader:
                output = self.learner(batch_x, batch_l)
                loss = self.loss_fn(output, batch_y)
                test_loss += loss.item()
                num_correct += torch.eq(
                    output.max(-1)[1], batch_y
                ).sum().item()
                num_total += batch_y.size()[0]
            test_accuracy = num_correct / num_total
            logger.info(
                (
                    'Dataset {}:\ttrain loss={:.5f}\ttest loss={:.5f}\t'
                    'train accuracy={:.5f}\ttest accuracy={:.5f}'
                ).format(
                    dataset, train_loss, test_loss,
                    train_accuracy, test_accuracy
                )
            )
            if test_loss < best_loss:
                patience = 0
                best_loss = test_loss
                torch.save(self.learner.state_dict(), model_path)
            else:
                patience += 1
                if patience == self.early_stopping:
                    break
        self.learner.load_state_dict(torch.load(model_path))
