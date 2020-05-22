import math

import higher

from models import utils
from models.base_models import ALBERTRelationModel
from torch import nn, optim
from torch.nn import functional as F

import coloredlogs
import logging
import os
import torch

from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class RelMetaModel(nn.Module):
    def __init__(self, config):
        super(RelMetaModel, self).__init__()
        self.base_path = config['base_path']
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.output_lr = config.get('output_lr', 0.1)

        if 'albert' in config['learner_model']:
            self.learner = ALBERTRelationModel(config['learner_params'])

        self.proto_maml = config.get('proto_maml', False)
        self.fomaml = config.get('fomaml', False)

        self.learner_loss = {}
        for task in config['learner_params']['num_outputs']:
            self.learner_loss[task] = nn.CrossEntropyLoss(ignore_index=-1)

        self.output_layer_weight = None
        self.output_layer_bias = None

        if config.get('trained_learner', False):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base_path, 'saved_models', config['trained_learner'])
            ))
            logger.info('Loaded trained learner model {}'.format(config['trained_learner']))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        if self.proto_maml:
            logger.info('Initialization of output layer weights as per prototypical networks turned on')

        params = [p for p in self.learner.parameters() if p.requires_grad]
        self.learner_optimizer = optim.SGD(params, lr=self.learner_lr)

    def move_to_device(self, batch_x, batch_y):
        for key in batch_x:
            batch_x[key] = batch_x[key].to(self.device)
        batch_y = batch_y.to(self.device)
        return batch_x, batch_y

    def forward(self, episodes, updates=1, testing=False):
        support_losses = []
        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):

            self.initialize_output_layer(episode.n_classes)

            if self.proto_maml:
                support_repr, support_label = [], []
                for batch_x, batch_y in episode.support_loader:
                    self.move_to_device(batch_x, batch_y)
                    output_repr = self.learner(batch_x)
                    support_repr.append(output_repr)
                    support_label.append(batch_y)
                init_weights, init_bias = self._initialize_with_proto_weights(support_repr, support_label, episode.n_classes)
            else:
                init_weights, init_bias = 0, 0

            with higher.innerloop_ctx(self.learner, self.learner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=(not self.fomaml and not testing)) as (flearner, diffopt):

                all_predictions, all_labels = [], []
                self.train()
                flearner.train()
                flearner.zero_grad()

                for i in range(updates):
                    for batch_x, batch_y in episode.support_loader:
                        batch_x, batch_y = self.move_to_device(batch_x, batch_y)
                        output = flearner(batch_x)
                        output = self.output_layer(output, init_weights, init_bias)
                        loss = self.learner_loss[episode.base_task](output, batch_y)

                        # Update the output layer parameters
                        output_weight_grad, output_bias_grad = torch.autograd.grad(loss, [self.output_layer_weight, self.output_layer_bias], retain_graph=True)
                        self.output_layer_weight = self.output_layer_weight - self.output_lr * output_weight_grad
                        self.output_layer_bias = self.output_layer_bias - self.output_lr * output_bias_grad

                        # Update the shared parameters
                        diffopt.step(loss, retain_graph=True)
                        loss = loss.detach()
                        output = output.detach()

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                pred = make_prediction(output[relevant_indices].detach()).cpu()
                all_predictions.extend(pred)
                all_labels.extend(batch_y[relevant_indices].cpu())

                support_loss = loss.item()

                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions, all_labels, binary=False)

                logger.info('Episode {}/{}, task {} [support_set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                            'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                        support_loss, accuracy, precision, recall, f1_score))

                query_loss = 0.0
                all_predictions, all_labels = [], []

                # Disable dropout
                for module in flearner.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()

                for n_batch, (batch_x, batch_y) in enumerate(episode.query_loader):
                    batch_x, batch_y = self.move_to_device(batch_x, batch_y)
                    output = flearner(batch_x)
                    output = self.output_layer(output, init_weights, init_bias)
                    loss = self.learner_loss[episode.base_task](output, batch_y)

                    if not testing:
                        if self.fomaml:
                            meta_grads = torch.autograd.grad(loss, [p for p in flearner.parameters() if p.requires_grad], retain_graph=self.proto_maml)
                        else:
                            meta_grads = torch.autograd.grad(loss, [p for p in flearner.parameters(time=0) if p.requires_grad], retain_graph=self.proto_maml)
                        if self.proto_maml:
                            proto_grads = torch.autograd.grad(loss, [p for p in self.learner.parameters() if p.requires_grad])
                            meta_grads = [mg + pg for (mg, pg) in zip(meta_grads, proto_grads)]
                    query_loss += loss.item()

                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    pred = make_prediction(output[relevant_indices].detach()).cpu()
                    all_predictions.extend(pred)
                    all_labels.extend(batch_y[relevant_indices].cpu())

                query_loss /= n_batch + 1

            accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions, all_labels, binary=False)

            logger.info('Episode {}/{}, task {} [query set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                    query_loss, accuracy, precision, recall, f1_score))
            support_losses.append(support_loss)
            query_losses.append(query_loss)
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1_score)

            if not testing:
                for param, meta_grad in zip([p for p in self.learner.parameters() if p.requires_grad], meta_grads):
                    if param.grad is not None:
                        param.grad += meta_grad.detach()
                    else:
                        param.grad = meta_grad.detach()

        # Average the accumulated gradients
        if not testing:
            for param in self.learner.parameters():
                if param.requires_grad:
                    param.grad /= len(query_accuracies)

        return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s

    def initialize_output_layer(self, n_classes):
        stdv = 1.0 / math.sqrt(self.learner.hidden_size)
        self.output_layer_weight = -2 * stdv * torch.rand((n_classes, self.learner.hidden_size),
                                                          device=self.device) + stdv
        self.output_layer_bias = -2 * stdv * torch.rand(n_classes, device=self.device) + stdv
        self.output_layer_weight.requires_grad = True
        self.output_layer_bias.requires_grad = True

    def _initialize_with_proto_weights(self, support_repr, support_label, n_classes):
        prototypes = self._build_prototypes(support_repr, support_label, n_classes)
        weight = 2 * prototypes
        bias = -torch.norm(prototypes, dim=1)**2
        self.output_layer_weight = torch.zeros_like(weight, requires_grad=True)
        self.output_layer_bias = torch.zeros_like(bias, requires_grad=True)
        return weight, bias

    def _build_prototypes(self, data_repr, data_label, num_outputs):
        n_dim = data_repr[0].shape[1]
        data_repr = torch.cat(tuple([x.view(-1, n_dim) for x in data_repr]), dim=0)
        data_label = torch.cat(tuple([y.view(-1) for y in data_label]), dim=0)

        prototypes = torch.zeros((num_outputs, n_dim), device=self.device)

        for c in range(num_outputs):
            idx = torch.nonzero(data_label == c).view(-1)
            if idx.nelement() != 0:
                prototypes[c] = torch.mean(data_repr[idx], dim=0)

        return prototypes

    def output_layer(self, input, weight, bias):
        return F.linear(input, self.output_layer_weight + weight, self.output_layer_bias + bias)
