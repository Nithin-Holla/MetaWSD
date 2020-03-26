import math

import higher
import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from transformers import BertTokenizer

from models import utils
from models.base_models import RNNSequenceModel, MLPModel, BERTSequenceModel
from torch import nn, optim
from torch.nn import functional as F

import coloredlogs
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
        self.output_lr = config.get('output_lr', 0.1)

        if 'seq' in config['learner_model']:
            self.learner = RNNSequenceModel(config['learner_params'])
        elif 'mlp' in config['learner_model']:
            self.learner = MLPModel(config['learner_params'])
        elif 'bert' in config['learner_model']:
            self.learner = BERTSequenceModel(config['learner_params'])

        self.proto_maml = config.get('proto_maml', False)
        self.fomaml = config.get('fomaml', False)
        self.vectors = config.get('vectors', 'glove')

        if self.vectors == 'elmo':
            self.elmo = Elmo(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                             weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                             num_output_representations=1,
                             dropout=0,
                             requires_grad=False)
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)
        elif self.vectors == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        self.learner_loss = {}
        for task in config['learner_params']['num_outputs']:
            if task == 'metaphor':
                self.learner_loss[task] = BCEWithLogitsLossAndIgnoreIndex(ignore_index=-1)
            else:
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

    def vectorize(self, batch_x, batch_len, batch_y):
        with torch.no_grad():
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
            elif self.vectors == 'bert':
                max_batch_len = max(batch_len) + 2
                input_ids = torch.zeros((len(batch_x), max_batch_len)).long()
                for i, sent in enumerate(batch_x):
                    sent_token_ids = self.bert_tokenizer.encode(sent, add_special_tokens=True)
                    input_ids[i, :len(sent_token_ids)] = torch.tensor(sent_token_ids)
                batch_x = input_ids.to(self.device)

        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def forward(self, episodes, updates=1, testing=False):
        support_losses = []
        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):

            self.initialize_output_layer(episode.n_classes)

            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)

            with torch.backends.cudnn.flags(enabled=self.fomaml or testing or not isinstance(self.learner, RNNSequenceModel)), \
                 higher.innerloop_ctx(self.learner, self.learner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=(not self.fomaml and not testing)) as (flearner, diffopt):

                all_predictions, all_labels = [], []
                self.train()
                flearner.train()
                flearner.zero_grad()

                for i in range(updates):
                    output = flearner(batch_x, batch_len)
                    if i == 0 and self.proto_maml:
                        self._initialize_with_proto_weights(output, batch_y, episode.n_classes)
                    output = self.output_layer(output)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss[episode.base_task](output, batch_y)

                    # Update the output layer parameters
                    output_weight_grad, output_bias_grad = torch.autograd.grad(loss, [self.output_layer_weight, self.output_layer_bias], retain_graph=True)
                    self.output_layer_weight.data -= self.output_lr * output_weight_grad
                    self.output_layer_bias.data -= self.output_lr * output_bias_grad

                    # Update the shared parameters
                    diffopt.step(loss)

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                pred = make_prediction(output[relevant_indices].detach()).cpu()
                all_predictions.extend(pred)
                all_labels.extend(batch_y[relevant_indices].cpu())

                support_loss = loss.item()

                if episode.base_task != 'metaphor':
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=False)
                else:
                    accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                    all_labels, binary=True)

                logger.info('Episode {}/{}, task {} [support_set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                            'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                        support_loss, accuracy, precision, recall, f1_score))

                query_loss = 0.0
                all_predictions, all_labels = [], []

                # Disable dropout
                for module in flearner.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()

                for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.query_loader):
                    batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                    output = flearner(batch_x, batch_len)
                    output = self.output_layer(output)
                    output = output.view(output.size()[0] * output.size()[1], -1)
                    batch_y = batch_y.view(-1)
                    loss = self.learner_loss[episode.base_task](output, batch_y)

                    if not testing:
                        if self.fomaml:
                            meta_grads = torch.autograd.grad(loss, [p for p in flearner.parameters() if p.requires_grad])
                        else:
                            meta_grads = torch.autograd.grad(loss, [p for p in flearner.parameters(time=0) if p.requires_grad])

                    query_loss += loss.item()

                    relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                    pred = make_prediction(output[relevant_indices].detach()).cpu()
                    all_predictions.extend(pred)
                    all_labels.extend(batch_y[relevant_indices].cpu())

                query_loss /= n_batch + 1

            if episode.base_task != 'metaphor':
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=False)
            else:
                accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                                all_labels, binary=True)

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

        if testing:
            return support_losses, query_accuracies, query_precisions, query_recalls, query_f1s
        else:
            return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s

    def initialize_output_layer(self, n_classes):
        if isinstance(self.learner, RNNSequenceModel):
            stdv = 1.0 / math.sqrt(self.learner.hidden_size // 4)
            self.output_layer_weight = -2 * stdv * torch.rand((n_classes, self.learner.hidden_size // 4), device=self.device) + stdv
            self.output_layer_bias = -2 * stdv * torch.rand(n_classes, device=self.device) + stdv
        elif isinstance(self.learner, MLPModel) or isinstance(self.learner, BERTSequenceModel):
            stdv = 1.0 / math.sqrt(self.learner.hidden_size)
            self.output_layer_weight = -2 * stdv * torch.rand((n_classes, self.learner.hidden_size),
                                                              device=self.device) + stdv
            self.output_layer_bias = -2 * stdv * torch.rand(n_classes, device=self.device) + stdv
        self.output_layer_weight.requires_grad = True
        self.output_layer_bias.requires_grad = True

    def _initialize_with_proto_weights(self, support_repr, support_label, n_classes):
        prototypes = self._build_prototypes(support_repr, support_label, n_classes)
        self.output_layer_weight = 2 * prototypes
        self.output_layer_bias = -torch.norm(prototypes, dim=1)

    def _build_prototypes(self, data_repr, data_label, num_outputs):
        n_dim = data_repr.shape[2]
        data_repr = data_repr.view(-1, n_dim)
        data_label = data_label.view(-1, n_dim)

        prototypes = torch.zeros((num_outputs, n_dim), device=self.device)

        for c in range(num_outputs):
            idx = torch.nonzero(data_label == c).view(-1)
            if idx.nelement() != 0:
                prototypes[c] = torch.mean(data_repr[idx], dim=0)

        return prototypes

    def output_layer(self, input):
        return F.linear(input, self.output_layer_weight, self.output_layer_bias)
