import coloredlogs
import logging
import torch
import torchtext
from allennlp.modules import Elmo
from allennlp.modules.elmo import batch_to_ids
from scipy.spatial.distance import cdist
from sklearn import metrics
import numpy as np
from transformers import BertTokenizer, BertModel

logger = logging.getLogger('NearestNeighbor Log')
coloredlogs.install(logger=logger, level='DEBUG',
                    fmt='%(asctime)s - %(name)s - %(levelname)s'
                        ' - %(message)s')


class NearestNeighborClassifier():
    def __init__(self, config):
        self.vectors = config.get('vectors', 'elmo')
        self.device = torch.device(config.get('device', 'cpu'))

        if self.vectors == 'elmo':
            self.elmo = Elmo(options_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json",
                             weight_file="https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5",
                             num_output_representations=1,
                             dropout=0,
                             requires_grad=False)
            self.elmo.to(self.device)
        elif self.vectors == 'glove':
            self.glove = torchtext.vocab.GloVe(name='840B', dim=300)
        elif self.vectors == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.bert = BertModel.from_pretrained('bert-base-cased')
            self.bert.to(self.device)

        logger.info('Nearest neighbor classifier instantiated')

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
                attention_mask = (batch_x.detach() != 0).float()
                batch_x, _ = self.bert(batch_x, attention_mask=attention_mask)
                batch_x = batch_x[:, 1:-1, :]
        batch_len = torch.tensor(batch_len).to(self.device)
        batch_y = torch.tensor(batch_y).to(self.device)
        return batch_x, batch_len, batch_y

    def training(self, train_episodes, val_episodes):
        return 0

    def testing(self, test_episodes):
        episode_accuracies, episode_precisions, episode_recalls, episode_f1s = [], [], [], []
        for episode_id, episode in enumerate(test_episodes):
            batch_x, batch_len, batch_y = next(iter(episode.support_loader))
            support_repr, _, support_labels = self.vectorize(batch_x, batch_len, batch_y)
            support_repr = support_repr.reshape(support_repr.shape[0] * support_repr.shape[1], -1)
            support_labels = support_labels.view(-1)
            support_repr = support_repr[support_labels != -1].cpu().numpy()
            support_labels = support_labels[support_labels != -1].cpu().numpy()

            batch_x, batch_len, batch_y = next(iter(episode.query_loader))
            query_repr, _, true_labels = self.vectorize(batch_x, batch_len, batch_y)
            query_repr = query_repr.reshape(query_repr.shape[0] * query_repr.shape[1], -1)
            true_labels = true_labels.view(-1)
            query_repr = query_repr[true_labels != -1].cpu().numpy()
            true_labels = true_labels[true_labels != -1].cpu().numpy()

            dist = cdist(query_repr, support_repr, metric='cosine')
            nearest_neighbor = np.argmin(dist, axis=1)
            predictions = support_labels[nearest_neighbor]

            accuracy = metrics.accuracy_score(true_labels, predictions)
            precision = metrics.precision_score(true_labels, predictions, average='macro')
            recall = metrics.recall_score(true_labels, predictions, average='macro')
            f1_score = metrics.f1_score(true_labels, predictions, average='macro')
            logger.info('Episode {}/{}, task {} [query set]: Accuracy = {:.5f}, precision = {:.5f}, '
                        'recall = {:.5f}, F1 score = {:.5f}'.format(episode_id + 1, len(test_episodes), episode.task_id,
                                                                    accuracy, precision, recall, f1_score))

            episode_accuracies.append(accuracy)
            episode_precisions.append(precision)
            episode_recalls.append(recall)
            episode_f1s.append(f1_score)

        logger.info('Avg meta-testing metrics: Accuracy = {:.5f}, precision = {:.5f}, recall = {:.5f}, '
                    'F1 score = {:.5f}'.format(np.mean(episode_accuracies),
                                               np.mean(episode_precisions),
                                               np.mean(episode_recalls),
                                               np.mean(episode_f1s)))

        return np.mean(episode_f1s)
