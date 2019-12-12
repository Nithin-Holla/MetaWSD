from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from models import utils
from models.base_models import RNNSequenceModel
from torch import nn
from torch import optim

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


class SeqBaselineModel(nn.Module):
    def __init__(self, config):
        super(SeqBaselineModel, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.learner = RNNSequenceModel(config['learner_params'])

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

        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_baseline'])
            ))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        self.elmo.to(self.device)

        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(params, lr=self.learner_lr, weight_decay=self.weight_decay)

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
            for _ in range(updates):
                self.train()
                support_loss = 0.0
                all_predictions, all_labels = [], []

                for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.support_loader):
                    batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                    output = self.learner(batch_x, batch_len)
                    output = self.output_layer[episode.task](output)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss[episode.task](output, batch_y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    support_loss += loss.item()

                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    pred = make_prediction(output[relevant_indices].detach()).cpu()
                    all_predictions.extend(pred)
                    all_labels.extend(batch_y[relevant_indices].cpu())

                self.optimizer.step()
                support_loss /= n_batch + 1

            if episode.task != 'metaphor':
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=False)
            else:
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=True)

            logger.info('Episode {}/{}, task {} [support_set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task,
                                                                    support_loss, accuracy, precision, recall,
                                                                    f1_score))

            query_loss = 0.0
            all_predictions, all_labels = [], []

            if testing:
                self.eval()

            for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.query_loader):
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                output = self.learner(batch_x, batch_len)
                output = self.output_layer[episode.task](output)
                output = output.view(output.size()[0] * output.size()[1], -1)
                batch_y = batch_y.view(-1)
                loss = self.learner_loss[episode.task](output, batch_y)

                if not testing:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

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

        if testing:
            return support_losses
        else:
            return query_losses, query_accuracies
