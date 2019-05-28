from base_models import RNNClassificationModel
from collections import OrderedDict
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNClassificationModel(
            config['learner_params'], config['embeddings'], embeds_grad=True
        )
        self.order = config['meta_order']
        if config['trained_learner']:
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'models', config['trained_learner'])
            ))

    def forward_order1(self, support_loaders, query_loaders, datasets, updates):
        query_losses = []
        accuracies = []
        for support, query, dataset in zip(
                support_loaders, query_loaders, datasets
        ):
            learner = copy.deepcopy(self.learner)
            optimizer = optim.SGD(learner.parameters(), lr=self.learner_lr)
            query_accuracy, query_loss = 0.0, 0.0
            for _ in range(updates):
                # Within-episode training on support set
                learner.train()
                optimizer.zero_grad()
                num_correct, num_total, support_loss = 0.0, 0.0, 0.0
                for batch_x, batch_y, batch_l in support:
                    output = learner(batch_x, batch_l)
                    loss = self.loss_fn(output, batch_y) / len(support)
                    loss.backward()
                    support_loss += loss.item()
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                support_accuracy = num_correct / num_total
                optimizer.step()
                # Within-episode pass on query set
                num_correct, num_total, query_loss = 0.0, 0.0, 0.0
                learner.zero_grad()
                learner.eval()
                for batch_x, batch_y, batch_l in query:
                    output = learner(batch_x, batch_l)
                    loss = self.loss_fn(output, batch_y) / len(query)
                    loss.backward()
                    query_loss += loss.item()
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                query_accuracy = num_correct / num_total
                logger.info(
                    (
                        'Dataset {}:\tsupport loss={:.5f}\tquery loss={:.5f}\t'
                        'support accuracy={:.5f}\tquery accuracy={:.5f}'
                    ).format(
                        dataset, support_loss, query_loss,
                        support_accuracy, query_accuracy
                    )
                )
            # Append the last query accuracy and query loss
            query_losses.append(query_loss)
            accuracies.append(query_accuracy)
            # Accumulate the gradients inside the meta learner
            for param, new_param in zip(
                self.learner.parameters(), learner.parameters()
            ):
                if param.grad is not None and param.requires_grad:
                    param.grad += new_param.grad
                elif param.requires_grad:
                    param.grad = new_param.grad
        # Average the accumulated gradients
        for param in self.learner.parameters():
            if param.requires_grad:
                param.grad /= len(accuracies)
        return query_losses, accuracies

    def forward_order2(self, support_loaders, query_loaders, datasets, updates):
        query_losses, query_accuracies = [], []
        for support, query, dataset in zip(
                support_loaders, query_loaders, datasets
        ):
            fast_weights = OrderedDict(
                (name, nn.Parameter(param))
                for (name, param) in self.learner.named_parameters()
            )
            query_accuracy, query_loss = 0.0, 0.0
            for _ in range(updates):
                # Within-episode training on support set
                self.learner.train()
                num_correct, num_total, support_loss = 0.0, 0.0, 0.0
                for batch_x, batch_y, batch_l in support:
                    output = self.learner(batch_x, batch_l, fast_weights)
                    support_loss += self.loss_fn(output, batch_y) / len(support)
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                support_accuracy = num_correct / num_total
                gradients = torch.autograd.grad(
                    support_loss, fast_weights.values(), create_graph=True
                )
                fast_weights = OrderedDict(
                    (name, nn.Parameter(param - self.learner_lr * grad))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
                # Within-episode pass on query set
                self.learner.eval()
                num_correct, num_total, query_loss = 0.0, 0.0, 0.0
                for batch_x, batch_y, batch_l in query:
                    output = self.learner(batch_x, batch_l, fast_weights)
                    query_loss += self.loss_fn(output, batch_y) / len(query)
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                query_accuracy = num_correct / num_total
                logger.info(
                    (
                        'Dataset {}:\tsupport loss={:.5f}\tquery loss={:.5f}\t'
                        'support accuracy={:.5f}\tquery accuracy={:.5f}'
                    ).format(
                        dataset, support_loss.item(), query_loss.item(),
                        support_accuracy, query_accuracy
                    )
                )
            (query_loss / len(datasets)).backward()
            query_losses.append(query_loss)
            query_accuracies.append(query_accuracy)
            # Accumulate the gradients inside the meta learner
            for param, new_param in zip(
                self.learner.parameters(), fast_weights.values()
            ):
                if param.grad is not None and param.requires_grad:
                    param.grad += new_param.grad
                elif param.requires_grad:
                    param.grad = new_param.grad
        return query_losses, query_accuracies

    def forward(self, support_loaders, query_loaders, datasets, updates=1):
        if self.order == 1:
            return self.forward_order1(support_loaders, query_loaders, datasets, updates)
        else:
            return self.forward_order2(support_loaders, query_loaders, datasets, updates)
