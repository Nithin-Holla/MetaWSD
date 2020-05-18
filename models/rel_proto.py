from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch
from transformers import AdamW, get_constant_schedule_with_warmup

from models import utils
from models.base_models import ALBERTRelationModel
from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class RelPrototypicalNetwork(nn.Module):
    def __init__(self, config):
        super(RelPrototypicalNetwork, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.lr = config.get('meta_lr', 1e-3)
        self.weight_decay = config.get('meta_weight_decay', 0.0)

        if 'albert' in config['learner_model']:
            self.learner = ALBERTRelationModel(config['learner_params'])

        self.num_outputs = config['learner_params']['num_outputs']

        self.loss_fn = {}
        for task in config['learner_params']['num_outputs']:
            self.loss_fn[task] = nn.CrossEntropyLoss(ignore_index=-1)

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        self.initialize_optimizer_scheduler()

    def initialize_optimizer_scheduler(self):
        learner_params = [p for p in self.learner.parameters() if p.requires_grad]
        if isinstance(self.learner, ALBERTRelationModel):
            self.optimizer = AdamW(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=100)
        else:
            self.optimizer = optim.Adam(learner_params, lr=self.lr, weight_decay=self.weight_decay)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)

    def move_to_device(self, batch_x, batch_y):
        for key in batch_x:
            batch_x[key] = batch_x[key].to(self.device)
        batch_y = batch_y.to(self.device)
        return batch_x, batch_y

    def forward(self, episodes, updates=1, testing=False):
        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):

            batch_x, batch_y = next(iter(episode.support_loader))
            batch_x, batch_y = self.move_to_device(batch_x, batch_y)

            self.train()

            batch_x_repr = self.learner(batch_x)

            prototypes = self._build_prototypes(batch_x_repr, batch_y, episode.n_classes)

            # Run on query
            query_loss = 0.0
            all_predictions, all_labels = [], []
            for module in self.learner.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            for n_batch, (batch_x, batch_y) in enumerate(episode.query_loader):
                batch_x, batch_y = self.move_to_device(batch_x, batch_y)
                batch_x_repr = self.learner(batch_x)
                output = self._normalized_distances(prototypes, batch_x_repr)
                loss = self.loss_fn[episode.base_task](output, batch_y)
                query_loss += loss.item()

                if not testing:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                all_predictions.extend(make_prediction(output[relevant_indices]).cpu())
                all_labels.extend(batch_y[relevant_indices].cpu())

            query_loss /= n_batch + 1

            # Calculate metrics

            accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions, all_labels, binary=False)
            logger.info('Episode {}/{}, task {} [query set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                    query_loss, accuracy, precision, recall, f1_score))

            query_losses.append(query_loss)
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1_score)

        return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s

    def _build_prototypes(self, data_repr, data_label, num_outputs):
        n_dim = data_repr.shape[1]

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
