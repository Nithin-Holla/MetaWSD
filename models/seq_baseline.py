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

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class SeqBaselineModel(nn.Module):
    def __init__(self, config):
        super(SeqBaselineModel, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.learner_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)
        self.learner = RNNSequenceModel(config['learner_params'])

        options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.elmo.requires_grad = False

        self.output_layer = {}
        for task in config['learner_params']['num_outputs']:
            self.output_layer[task] = nn.Linear(self.learner.hidden // 2, config['learner_params']['num_outputs'][task])
        self.output_layer = nn.ModuleDict(self.output_layer)

        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_baseline'])
            ))

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.elmo.to(self.device)

        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(learner_params, lr=self.learner_lr, weight_decay=self.weight_decay)

    def vectorize(self, batch_x, batch_y):
        with torch.no_grad():
            char_ids = batch_to_ids(batch_x)
            char_ids = char_ids.to(self.device)
            batch_x = self.elmo(char_ids)['elmo_representations'][0]
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_y

    def forward(self, episodes, epochs=1):
        best_loss = float('inf')
        best_model = None
        patience = 0

        for epoch in range(epochs):
            for episode in episodes:
                for batch_x, batch_y in episode.support_loader:
                    batch_x, batch_y = self.vectorize(batch_x, batch_y)
                    output = self.learner(batch_x)
                    output = self.output_layer[episode.task](output)
                    loss = self.learner_loss(output.view(output.size()[0] * output.size()[1], -1), batch_y.view(-1))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss = 0.0
                all_predictions, all_labels = [], []
                for batch_x, batch_y in episode.query_loader:
                    batch_x, batch_y = self.vectorize(batch_x, batch_y)
                    output = self.learner(batch_x)
                    output = self.output_layer[episode.task](output)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss(output, batch_y)
                    total_loss += loss.item()
                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    all_predictions.extend(output[relevant_indices].max(-1)[1])
                    all_labels.extend(batch_y[relevant_indices])

                if episode.task != 'metaphor':
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=False)
                else:
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=True)

                logger.info('Task {}: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                            'F1 score = {:.5f}'.format(episode.task, total_loss, accuracy, precision,
                                                       recall, f1_score))

                if total_loss < best_loss:
                    patience = 0
                    best_loss = total_loss
                    best_model = copy.deepcopy(self.learner)
                else:
                    patience += 1
                    if patience == self.early_stopping:
                        break
        self.learner = copy.deepcopy(best_model)
