from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from models import utils
from models.base_models import RNNSequenceModel
from torch import nn, optim

import coloredlogs
import copy
import logging
import os
import torch

from models.loss import BCEWithLogitsLossAndIgnoreIndex
from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class SeqMetaModel(nn.Module):
    def __init__(self, config):
        super(SeqMetaModel, self).__init__()
        self.base_path = config['base_path']
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.learner = RNNSequenceModel(config['learner_params'])
        self.num_outputs = config['learner_params']['num_outputs']
        self.num_episodes = config['num_episodes']
        self.proto_maml = config['proto_maml']

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.elmo.requires_grad = False

        self.output_layer, self.learner_loss = {}, {}
        for task in config['learner_params']['num_outputs']:
            self.output_layer[task] = nn.Linear(self.learner.hidden // 2, config['learner_params']['num_outputs'][task])
            if task == 'metaphor':
                self.learner_loss[task] = BCEWithLogitsLossAndIgnoreIndex(ignore_index=-1)
            else:
                self.learner_loss[task] = nn.CrossEntropyLoss(ignore_index=-1)
        self.output_layer = nn.ModuleDict(self.output_layer)
        self.learner_loss = nn.ModuleDict(self.learner_loss)

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        self.elmo.to(self.device)

        if self.proto_maml:
            logger.info('Initialization of output layer weights as per prototypical networks turned on')

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            char_ids = batch_to_ids(batch_x)
            char_ids = char_ids.to(self.device)
            batch_x = self.elmo(char_ids)['elmo_representations'][0]
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def forward(self, episodes, updates=1, testing=False):
        support_losses, query_losses, query_accuracies = [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):
            learner = copy.deepcopy(self.learner)
            params = [p for p in learner.parameters() if p.requires_grad] + \
                     [p for p in self.output_layer[episode.task].parameters() if p.requires_grad]
            learner_optimizer = optim.SGD(params, lr=self.learner_lr)

            if self.proto_maml:
                self._initialize_with_proto_weights(episode.support_loader, episode.task)

            for _ in range(updates):
                self.train()
                learner.train()
                learner_optimizer.zero_grad()
                support_loss = 0.0
                all_predictions, all_labels = [], []

                for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.support_loader):
                    batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                    output = learner(batch_x, batch_len)
                    output = self.output_layer[episode.task](output)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss[episode.task](output, batch_y)
                    loss.backward()
                    support_loss += loss.item()

                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    pred = make_prediction(output[relevant_indices].detach()).cpu()
                    all_predictions.extend(pred)
                    all_labels.extend(batch_y[relevant_indices].cpu())

                learner_optimizer.step()
                support_loss /= n_batch + 1

            if episode.task != 'metaphor':
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=False)
            else:
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=True)

            logger.info('Episode {}/{}, task {} [support_set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task,
                                                                    support_loss, accuracy, precision, recall, f1_score))

            query_loss = 0.0
            all_predictions, all_labels = [], []
            learner_optimizer.zero_grad()

            if testing:
                self.eval()
                learner.eval()

            for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.query_loader):
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                output = learner(batch_x, batch_len)
                output = self.output_layer[episode.task](output)
                output = output.view(output.size()[0] * output.size()[1], -1)
                batch_y = batch_y.view(-1)
                loss = self.learner_loss[episode.task](output, batch_y)

                if not testing:
                    loss.backward()

                query_loss += loss.item()

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                pred = make_prediction(output[relevant_indices].detach()).cpu()
                all_predictions.extend(pred)
                all_labels.extend(batch_y[relevant_indices].cpu())

            query_loss /= n_batch + 1

            if episode.task != 'metaphor':
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=False)
            else:
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=True)

            logger.info('Episode {}/{}, task {} [query set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task,
                                                                    query_loss, accuracy, precision, recall, f1_score))
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            query_accuracies.append(accuracy)

            if not testing:
                for param, new_param in zip(
                    self.learner.parameters(), learner.parameters()
                ):
                    if param.grad is not None and param.requires_grad:
                        param.grad += new_param.grad
                    elif param.requires_grad:
                        param.grad = new_param.grad

        # Average the accumulated gradients
        if not testing:
            for param in self.learner.parameters():
                if param.requires_grad:
                    param.grad /= len(query_accuracies)

        if testing:
            return support_losses
        else:
            return query_losses, query_accuracies

    def _initialize_with_proto_weights(self, support_loader, task):
        support_repr, support_label = [], []
        for batch_x, batch_len, batch_y in support_loader:
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
            batch_x_repr = self.learner(batch_x, batch_len)
            support_repr.append(batch_x_repr)
            support_label.append(batch_y)

        prototypes = self._build_prototypes(support_repr, support_label, self.num_outputs[task])

        self.output_layer[task].weight.data = 2 * prototypes
        self.output_layer[task].bias.data = torch.norm(prototypes, dim=1)

    def _build_prototypes(self, data_repr, data_label, num_outputs):
        n_dim = data_repr[0].shape[2]
        data_repr = torch.cat(tuple([x.view(-1, n_dim) for x in data_repr]), dim=0)
        data_label = torch.cat(tuple([y.view(-1) for y in data_label]), dim=0)

        prototypes = torch.zeros((num_outputs, n_dim), device=self.device)

        for c in range(num_outputs):
            idx = torch.nonzero(data_label == c).view(-1)
            if idx.nelement() != 0:
                prototypes[c] = torch.mean(data_repr[idx], dim=0)

        return prototypes
