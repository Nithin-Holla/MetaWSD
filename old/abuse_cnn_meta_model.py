from old.base_models import CNNClassificationModel
from collections import OrderedDict
from torch import nn

import coloredlogs
import logging
import os
import torch

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class AbuseCNNMetaModel(nn.Module):
    def __init__(self, config):
        super(AbuseCNNMetaModel, self).__init__()
        self.base = config['base']
        self.embeddings_file = config['embeddings']
        self.loss_fn = nn.CrossEntropyLoss()
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = CNNClassificationModel(
            config['learner_params'], config['embeddings'], embeds_grad=True
        )
        self.order = config['meta_order']
        if config['trained_learner']:
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_learner'])
            ))

    def forward(self, support_loaders, query_loaders, datasets, updates=1):
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
                num_correct, num_total, support_loss = 0.0, 0.0, 0.0
                for batch_x, batch_y, batch_l in support:
                    output = self.learner(batch_x, fast_weights)
                    support_loss += self.loss_fn(output, batch_y) / len(support)
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                support_accuracy = num_correct / num_total
                gradients = torch.autograd.grad(
                    support_loss, fast_weights.values(),
                    create_graph=(self.order == 2)
                )
                fast_weights = OrderedDict(
                    (name, nn.Parameter(param - self.learner_lr * grad))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
                # Within-episode pass on query set
                num_correct, num_total, query_loss = 0.0, 0.0, 0.0
                for batch_x, batch_y, batch_l in query:
                    output = self.learner(batch_x, fast_weights, train=False)
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
