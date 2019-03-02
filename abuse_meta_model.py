from base_models import RNNClassificationModel
from torch import nn
from torch import optim

import coloredlogs
import copy
import logging
import os
import torch

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class AbuseMetaModel(nn.Module):
    def __init__(self, config):
        super(AbuseMetaModel, self).__init__()
        self.base = config['base']
        self.embeddings_file = config['embeddings']
        self.learner_loss = nn.CrossEntropyLoss()
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNClassificationModel(
            config['learner_params'], config['embeddings']
        )
        if config['trained_learner']:
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_learner'])
            ))

    def forward(self, support_loaders, query_loaders, languages, updates=1):
        query_losses = []
        accuracies = []
        for support, query, lang in zip(
                support_loaders, query_loaders, languages
        ):
            learner = copy.deepcopy(self.learner)
            optimizer = optim.SGD(learner.parameters(), lr=self.learner_lr)
            num_correct, num_total, query_loss = 0, 0, 0.0
            for _ in range(updates):
                for batch_x, batch_y in support:
                    output = learner(batch_x)
                    support_loss = self.learner_loss(output, batch_y)
                    optimizer.zero_grad()
                    support_loss.backward()
                    optimizer.step()
                    print(support_loss.item())

                num_correct, num_total, query_loss = 0, 0, 0.0
                learner.zero_grad()
                for batch_x, batch_y in query:
                    output = learner(batch_x)
                    loss = self.learner_loss(output, batch_y)
                    loss.backward()
                    query_loss += loss.item()
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                logger.info('Dataset {}: loss = {:.5f} accuracy = {:.5f}'.format(
                    lang, query_loss, 1.0 * num_correct / num_total
                ))
            query_losses.append(query_loss)
            accuracies.append(1.0 * num_correct / num_total)

            for param, new_param in zip(
                self.learner.parameters(), learner.parameters()
            ):
                if param.grad is not None and param.requires_grad:
                    param.grad += new_param.grad
                elif param.requires_grad:
                    param.grad = new_param.grad
        return query_losses, accuracies
