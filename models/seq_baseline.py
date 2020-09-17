import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from transformers import BertTokenizer, get_constant_schedule_with_warmup, AdamW

from models import utils
from models.base_models import RNNSequenceModel, MLPModel, BERTSequenceModel
from torch import nn
from torch import optim

import coloredlogs
import logging
import os
import torch

from models.utils import make_prediction

logger = logging.getLogger('Log')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SeqBaselineModel(nn.Module):
    def __init__(self, config):
        super(SeqBaselineModel, self).__init__()
        self.base_path = config['base_path']
        self.early_stopping = config['early_stopping']
        self.learner_lr = config.get('learner_lr', 1e-3)
        self.output_lr = config.get('output_lr', 0.1)
        self.weight_decay = config.get('meta_weight_decay', 0.0)

        if 'seq' in config['learner_model']:
            self.learner = RNNSequenceModel(config['learner_params'])
        elif 'mlp' in config['learner_model']:
            self.learner = MLPModel(config['learner_params'])
        elif 'bert' in config['learner_model']:
            self.learner = BERTSequenceModel(config['learner_params'])

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
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-large-cased')

        self.learner_loss = {}
        for task in config['learner_params']['num_outputs']:
            self.learner_loss[task] = nn.CrossEntropyLoss(ignore_index=-1)

        self.output_layer = None

        if config.get('trained_baseline', None):
            self.learner.load_state_dict(torch.load(
                os.path.join(self.base, 'saved_models', config['trained_baseline'])
            ))
            logger.info('Loaded trained baseline model {}'.format(config['trained_baseline']))

        self.device = torch.device(config.get('device', 'cpu'))
        self.to(self.device)

        self.learner_optimizer = None
        self.lr_scheduler = None
        self.initialize_optimizer_scheduler()

    def initialize_optimizer_scheduler(self):
        learner_params = [p for n, p in self.named_parameters() if p.requires_grad]
        if isinstance(self.learner, BERTSequenceModel):
            self.learner_optimizer = AdamW(learner_params, lr=self.learner_lr, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule_with_warmup(self.learner_optimizer, num_warmup_steps=100)
        else:
            self.learner_optimizer = optim.Adam(learner_params, lr=self.learner_lr, weight_decay=self.weight_decay)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.learner_optimizer, step_size=5000, gamma=0.5)

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

        if not testing:
            updates = 1
        else:
            # Learner optimizer
            learner_params = [p for n, p in self.named_parameters() if p.requires_grad and 'output_layer' not in n]
            finetune_learner_optimizer = optim.SGD(learner_params, lr=self.learner_lr, weight_decay=self.weight_decay)

            # Output optimizer
            output_params = [p for n, p in self.named_parameters() if p.requires_grad and 'output_layer' in n]
            output_optimizer = optim.SGD(output_params, lr=self.output_lr)

        support_losses = []
        query_losses, query_accuracies, query_precisions, query_recalls, query_f1s = [], [], [], [], []
        n_episodes = len(episodes)

        for episode_id, episode in enumerate(episodes):

            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
            episode_unique_labels = torch.unique(batch_y.view(-1)[batch_y.view(-1) != -1])

            self.train()

            all_predictions, all_labels = [], []

            for _ in range(updates):
                output = self.learner(batch_x, batch_len)
                output = output.view(output.size()[0] * output.size()[1], -1)
                batch_y = batch_y.view(-1)
                output = utils.subset_softmax(output, episode_unique_labels)
                loss = self.learner_loss[episode.base_task](output, batch_y)
                if testing:
                    output_optimizer.zero_grad()
                    finetune_learner_optimizer.zero_grad()
                else:
                    self.learner_optimizer.zero_grad()
                loss.backward()
                if testing:
                    output_optimizer.step()
                    finetune_learner_optimizer.step()
                else:
                    self.learner_optimizer.step()
                    self.lr_scheduler.step()

            relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
            pred = make_prediction(output[relevant_indices].detach()).cpu()
            all_predictions.extend(pred)
            all_labels.extend(batch_y[relevant_indices].cpu())

            support_loss = loss.item()

            accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                            all_labels, binary=False)

            logger.info('Episode {}/{}, task {} [support_set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                    support_loss, accuracy, precision, recall,
                                                                    f1_score))

            query_loss = 0.0
            all_predictions, all_labels = [], []

            if testing:
                self.eval()

            for n_batch, (batch_x, batch_len, batch_y) in enumerate(episode.query_loader):
                batch_x, batch_len, batch_y = self.vectorize(batch_x, batch_len, batch_y)
                output = self.learner(batch_x, batch_len)
                output = output.view(output.size()[0] * output.size()[1], -1)
                batch_y = batch_y.view(-1)
                output = utils.subset_softmax(output, episode_unique_labels)
                loss = self.learner_loss[episode.base_task](output, batch_y)

                if not testing:
                    self.learner_optimizer.zero_grad()
                    loss.backward()
                    self.learner_optimizer.step()
                    self.lr_scheduler.step()

                query_loss += loss.item()

                relevant_indices = torch.nonzero(batch_y != -1).view(-1).detach()
                pred = make_prediction(output[relevant_indices].detach()).cpu()
                all_predictions.extend(pred)
                all_labels.extend(batch_y[relevant_indices].cpu())

            query_loss /= n_batch + 1

            accuracy, precision, recall, f1_score = utils.calculate_metrics(all_predictions,
                                                                            all_labels, binary=False)

            logger.info('Episode {}/{}, task {} [query set]: Loss = {:.5f}, accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, n_episodes, episode.task_id,
                                                                    query_loss, accuracy, precision, recall, f1_score))

            support_losses.append(support_loss)
            query_losses.append(query_loss)
            query_accuracies.append(accuracy)
            query_precisions.append(precision)
            query_recalls.append(recall)
            query_f1s.append(f1_score)

        if testing:
            return support_losses, query_accuracies, query_precisions, query_recalls, query_f1s
        else:
            return query_losses, query_accuracies, query_precisions, query_recalls, query_f1s
