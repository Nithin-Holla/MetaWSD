import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids

from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch

from models import utils
from models.base_models import RNNSequenceModel, MLPModel
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

        if 'seq' in config['learner_model']:
            self.learner = RNNSequenceModel(config['learner_params'])
        elif 'mlp' in config['learner_model']:
            self.learner = MLPModel(config['learner_params'])

        self.num_outputs = config['learner_params']['num_outputs']
        self.vectors = config.get('vectors', 'glove')

        if self.vectors == 'elmo':
            self.elmo = Elmo(
                options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
                weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
                num_output_representations=1,
                dropout=0,
                requires_grad=False)
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)

        self.loss_fn = {}
        for task in config['learner_params']['num_outputs']:
            if task == 'metaphor':
                self.loss_fn[task] = BCEWithLogitsLossAndIgnoreIndex(ignore_index=-1)
            else:
                self.loss_fn[task] = nn.CrossEntropyLoss(ignore_index=-1)

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))

        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(learner_params, lr=self.lr, weight_decay=self.weight_decay)

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        if self.vectors == 'elmo':
            self.elmo.to(self.device)

    def vectorize(self, batch_x, batch_len, batch_y):
        if self.vectors == 'elmo':
            char_ids = batch_to_ids(batch_x)
            char_ids = char_ids.to(self.device)
            batch_x = self.elmo(char_ids)['elmo_representations'][0]
        elif self.vectors == 'glove':
            max_batch_len = max(batch_len)
            vec_batch_x = torch.ones((len(batch_x), max_batch_len, 300))
            for i, sent in enumerate(batch_x):
                sent_emb = self.glove.get_vecs_by_tokens(sent, lower_case_backup=True)
                vec_batch_x[i, :len(sent_emb)] = sent_emb
            batch_x = vec_batch_x.to(self.device)
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def forward(self, episodes, updates=1, testing=False):
        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):

            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
            episode_unique_labels = torch.unique(batch_y.view(-1)[batch_y.view(-1) != -1])

            self.train()
            support_repr, support_label = [], []

            batch_x_repr = self.learner(batch_x, batch_len)
            support_repr.append(batch_x_repr)
            support_label.append(batch_y)

            prototypes = self._build_prototypes(support_repr, support_label, episode.n_classes)

            # Run on query
            query_loss = 0.0
            all_predictions, all_labels = [], []

            if testing:
                self.eval()

            for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.query_loader):
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                batch_x_repr = self.learner(batch_x, batch_len)
                output = self._normalized_distances(prototypes, batch_x_repr)
                output = output.view(output.size()[0] * output.size()[1], -1)
                batch_y = batch_y.view(-1)
                output = utils.subset_softmax(output, episode_unique_labels)
                loss = self.loss_fn[episode.base_task](output, batch_y)
                query_loss += loss.item()

                if not testing:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                all_predictions.extend(make_prediction(output[relevant_indices]).cpu())
                all_labels.extend(batch_y[relevant_indices].cpu())

            query_loss /= n_batch + 1

            # Calculate metrics
            if episode.base_task != 'metaphor':
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=False)
            else:
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=True)

            logger.info('Episode {}/{}, task {} [query set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                    query_loss, accuracy, precision, recall, f1_score))

            query_losses.append(query_loss)
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1_score)

        return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s

    def _build_prototypes(self, data_repr, data_label, num_outputs, subset_classes=None):
        n_dim = data_repr[0].shape[2]
        data_repr = torch.cat(tuple([x.view(-1, n_dim) for x in data_repr]), dim=0)
        data_label = torch.cat(tuple([y.view(-1) for y in data_label]), dim=0)

        prototypes = torch.zeros((num_outputs, n_dim), device=self.device)

        if subset_classes is None or len(subset_classes) == 0:
            class_prototypes_required = range(num_outputs)
        else:
            class_prototypes_required = subset_classes

        for c in class_prototypes_required:
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
