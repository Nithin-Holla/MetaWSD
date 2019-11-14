from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from models.base_models import RNNSequenceModel
from torch import nn

import coloredlogs
import copy
import logging
import os
import torch

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class SeqMetaModel(nn.Module):
    def __init__(self, config):
        super(SeqMetaModel, self).__init__()
        self.base_path = config['base_path']
        self.learner_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNSequenceModel(config['learner_params'])
        self.num_outputs = config['learner_params']['num_outputs']
        self.num_episodes = config['num_episodes']

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.elmo.to(self.device)

        self.output_layer = {}
        for task in config['learner_params']['num_outputs']:
            self.output_layer[task] = nn.Linear(self.learner.hidden // 2, config['learner_params']['num_outputs'][task])
            self.output_layer[task] = self.output_layer[task].to(self.device)

        if config['trained_learner']:
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))

    def vectorize(self, batch_x, batch_y):
        char_ids = batch_to_ids(batch_x)
        char_ids = char_ids.to(self.device)
        batch_x = self.elmo(char_ids)['elmo_representations'][0]
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_y

    def forward(self, episodes, updates=1):
        query_losses = []
        accuracies = []
        for episode in episodes:
            learner = copy.deepcopy(self.learner)
            num_correct, num_total, query_loss = 0, 0, 0.0
            for _ in range(updates):
                for batch_x, batch_y in episode.support_loader:
                    batch_x, batch_y = self.vectorize(batch_x, batch_y)
                    output = learner(batch_x)
                    output = self.output_layer[episode.task](output)
                    loss = self.learner_loss(
                        output.view(output.size()[0] * output.size()[1], -1),
                        batch_y.view(-1)
                    )
                    params = [
                        p for p in learner.parameters() if p.requires_grad
                    ]
                    grads = torch.autograd.grad(loss, params)
                    for param, grad in zip(params, grads):
                        param.data -= grad * self.learner_lr

                num_correct, num_total, query_loss = 0, 0, 0.0
                learner.zero_grad()
                for batch_x, batch_y in episode.query_loader:
                    batch_x, batch_y = self.vectorize(batch_x, batch_y)
                    output = learner(batch_x)
                    output = self.output_layer[episode.task](output)
                    output = output.view(
                        output.size()[0] * output.size()[1], -1
                    )
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss(output, batch_y)
                    loss.backward()
                    query_loss += loss.item()
                    num_correct += torch.eq(
                        output.max(-1)[1], batch_y
                    ).sum().item()
                    num_total += batch_y.size()[0]
                logger.info('Task {}: loss = {:.5f} accuracy = {:.5f}'.format(
                    episode.task, query_loss, 1.0 * num_correct / num_total
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
        # Average the accumulated gradients
        for param in self.learner.parameters():
            if param.requires_grad:
                param.grad /= len(accuracies)
        return query_losses, accuracies
