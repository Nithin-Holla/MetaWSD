from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch

from models import utils
from models.base_models import RNNSequenceModel
from models.loss import BCEWithLogitsLossAndIgnoreIndex
from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class SeqPrototypicalNetwork(nn.Module):
    def __init__(self, config):
        super(SeqPrototypicalNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.learner = RNNSequenceModel(config['learner_params'])
        self.num_outputs = config['learner_params']['num_outputs']
        self.num_episodes = config['num_episodes']

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.elmo.requires_grad = False

        self.loss_fn = {}
        for task in config['learner_params']['num_outputs']:
            if task == 'metaphor':
                self.loss_fn[task] = BCEWithLogitsLossAndIgnoreIndex(ignore_index=-1)
            else:
                self.loss_fn[task] = nn.CrossEntropyLoss(ignore_index=-1)
        self.loss_fn = nn.ModuleDict(self.loss_fn)

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_learner'])
            ))

        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(learner_params, lr=self.lr, weight_decay=self.weight_decay)

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)
        self.elmo.to(self.device)

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
            char_ids = batch_to_ids(batch_x)
            char_ids = char_ids.to(self.device)
            batch_x = self.elmo(char_ids)['elmo_representations'][0]
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def forward(self, episodes, updates=1, testing=False):
        query_losses = []
        query_accuracies = []

        for episode in episodes:
            query_loss = 0.0
            for epoch in range(updates):
                self.train()
                support_repr, support_label = [], []
                for batch_x, batch_len, batch_y in episode.support_loader:
                    batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                    batch_x_repr = self.learner(batch_x, batch_len)
                    support_repr.append(batch_x_repr)
                    support_label.append(batch_y)

                prototypes = self._build_prototypes(support_repr, support_label, self.num_outputs[episode.task])

                # Run on query
                query_loss = 0.0
                all_predictions, all_labels = [], []

                if testing:
                    self.eval()

                for batch_x, batch_len, batch_y in episode.query_loader:
                    batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                    batch_x_repr = self.learner(batch_x, batch_len)
                    output = self._normalized_distances(prototypes, batch_x_repr)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.loss_fn[episode.task](output, batch_y)
                    query_loss += loss.item()

                    if not testing:
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        self.optimizer.step()

                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    all_predictions.extend(make_prediction(output[relevant_indices]).cpu())
                    all_labels.extend(batch_y[relevant_indices].cpu())

                # Calculate metrics
                if episode.task != 'metaphor':
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=False)
                else:
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=True)

                logger.info('Task {}: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                            'F1 score = {:.5f}'.format(episode.task, query_loss, accuracy, precision,
                                                       recall, f1_score))

            query_losses.append(query_loss)
            query_accuracies.append(accuracy)

        return query_losses, query_accuracies

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

    def _normalized_distances(self, prototypes, q):
        d = torch.stack(
            tuple([q.sub(p).pow(2).sum(dim=-1) for p in prototypes]),
            dim=-1
        )
        return d.neg()
